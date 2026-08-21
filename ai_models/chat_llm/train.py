"""
Fine-Tuning Script for Sarathi Legal Chat LLM

Uses Unsloth + LoRA to fine-tune a base model on Indian legal conversations.
Unsloth gives 2-5x faster training with 70% less memory than standard HF.

Hardware Requirements:
  - Minimum: 8GB VRAM (RTX 3060, RTX 4060, etc.) for 4-bit quantized 8B model
  - Recommended: 16GB VRAM (RTX 4080, A4000) for better batch sizes
  - CPU-only: Possible but very slow (hours instead of minutes)

Usage:
    # First, prepare the data
    python prepare_data.py

    # Then fine-tune
    python train.py

    # Fine-tune with custom settings
    python train.py --base-model unsloth/llama-3.1-8b-bnb-4bit --epochs 5 --lr 2e-4

    # Export to Ollama after training
    python train.py --export-only --output-dir ./sarathi-legal-lora
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("sarathi.train")

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "training_data" / "processed"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "sarathi-legal-lora"


def load_training_data(data_dir: Path):
    """Load the processed training data into HuggingFace datasets."""
    from datasets import load_dataset

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. Run prepare_data.py first."
        )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(val_path) if val_path.exists() else None,
        },
    )

    logger.info("Loaded %d training, %d validation examples",
                len(dataset["train"]),
                len(dataset.get("validation", [])))
    return dataset


def setup_model(base_model: str, max_seq_length: int, lora_rank: int):
    """
    Load the base model with 4-bit quantization and attach LoRA adapters.

    LoRA Target Modules:
      - q_proj, k_proj, v_proj: Attention projections (core reasoning)
      - o_proj: Output projection
      - gate_proj, up_proj, down_proj: MLP layers (factual knowledge)
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect (float16 on GPU, float32 on CPU)
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank * 2,  # Standard practice: alpha = 2 * rank
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable parameters: %d / %d (%.2f%%)",
                trainable, total, 100 * trainable / total)

    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    return model, tokenizer


def format_conversation(example, tokenizer):
    """
    Format a single training example into the model's chat template.

    Converts the {"conversations": [...]} format into the tokenizer's
    expected chat format with proper special tokens.
    """
    messages = []
    for turn in example["conversations"]:
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        messages.append({
            "role": role_map.get(turn["from"], turn["from"]),
            "content": turn["value"],
        })

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


def train(
    base_model: str,
    max_seq_length: int,
    lora_rank: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_dir: Path,
):
    """Run the full fine-tuning pipeline."""
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Load data
    dataset = load_training_data(DATA_DIR)

    # Setup model
    logger.info("Loading base model: %s", base_model)
    model, tokenizer = setup_model(base_model, max_seq_length, lora_rank)

    # Format conversations into model's chat template
    logger.info("Formatting training data...")
    train_dataset = dataset["train"].map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=dataset["train"].column_names,
    )
    val_dataset = None
    if "validation" in dataset:
        val_dataset = dataset["validation"].map(
            lambda x: format_conversation(x, tokenizer),
            remove_columns=dataset["validation"].column_names,
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        save_total_limit=2,
        seed=42,
        report_to="none",  # Set to "wandb" if you use Weights & Biases
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=True,  # Pack short examples together for efficiency
    )

    logger.info("Starting training...")
    trainer.train()

    # Save the LoRA adapter
    logger.info("Saving LoRA adapter to %s", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info("✅ Training complete!")
    return model, tokenizer


def export_to_gguf(output_dir: Path, quantization: str = "q4_k_m"):
    """
    Export the fine-tuned model to GGUF format for Ollama.

    After export, load into Ollama with:
        ollama create sarathi-legal -f Modelfile
    where the Modelfile's FROM points to the exported .gguf file.
    """
    from unsloth import FastLanguageModel

    logger.info("Loading trained model from %s", output_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(output_dir),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    gguf_dir = output_dir / "gguf"
    logger.info("Exporting to GGUF (%s quantization)...", quantization)
    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method=quantization,
    )

    logger.info("✅ GGUF export complete! File saved to %s", gguf_dir)
    logger.info(
        "\nTo load in Ollama:\n"
        "  1. Update the Modelfile: FROM %s/unsloth.%s.gguf\n"
        "  2. Run: ollama create sarathi-legal -f Modelfile\n"
        "  3. Test: ollama run sarathi-legal",
        gguf_dir,
        quantization.upper().replace("_", "_"),
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Sarathi Legal Chat LLM")

    parser.add_argument(
        "--base-model", default="unsloth/llama-3.1-8b-bnb-4bit",
        help="Base model from HuggingFace (default: Llama 3.1 8B 4-bit)",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank (higher = more capacity, more VRAM)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--export-only", action="store_true",
                        help="Skip training, only export existing model to GGUF")
    parser.add_argument("--quantization", default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="GGUF quantization level")

    args = parser.parse_args()

    if args.export_only:
        export_to_gguf(args.output_dir, args.quantization)
    else:
        train(
            base_model=args.base_model,
            max_seq_length=args.max_seq_length,
            lora_rank=args.lora_rank,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            output_dir=args.output_dir,
        )

        # Automatically export to GGUF
        export_to_gguf(args.output_dir, args.quantization)


if __name__ == "__main__":
    main()
