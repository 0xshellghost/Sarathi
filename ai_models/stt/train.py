"""
Fine-Tuning Script for Sarathi Speech-to-Text Model

Fine-tunes OpenAI's Whisper (open-source) on Hindi-English legal audio data
using HuggingFace's transformers + datasets libraries.

Whisper is inherently multilingual and handles Hindi well out of the box.
Fine-tuning improves accuracy on legal terminology and Indian accents.

Hardware Requirements:
  - Minimum: 8GB VRAM for whisper-small
  - Recommended: 16GB VRAM for whisper-medium
  - CPU: Feasible for whisper-tiny/base but very slow

Data Requirements:
  Put your audio training data in training_data/ with the structure:
    training_data/
    ├── metadata.csv       # Columns: file_name,transcription,language
    ├── audio/
    │   ├── sample_001.wav
    │   ├── sample_002.wav
    │   └── ...

  metadata.csv example:
    file_name,transcription,language
    sample_001.wav,"My landlord is not returning security deposit",en
    sample_002.wav,"Mera makan malik deposit wapas nahi kar raha",hi

Usage:
    python train.py
    python train.py --model openai/whisper-small --epochs 5
    python train.py --eval-only --model ./output/sarathi-stt
"""

import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("sarathi.stt.train")

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "training_data"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "sarathi-stt"


def load_audio_dataset(data_dir: Path):
    """
    Load audio dataset from metadata.csv and audio files.

    Expected structure:
        data_dir/
        ├── metadata.csv
        └── audio/
            ├── sample_001.wav
            └── ...
    """
    from datasets import Dataset, Audio
    import csv

    metadata_path = data_dir / "metadata.csv"
    audio_dir = data_dir / "audio"

    if not metadata_path.exists():
        logger.warning(
            "No training data found at %s.\n"
            "To train the STT model, create:\n"
            "  1. %s/metadata.csv with columns: file_name,transcription,language\n"
            "  2. %s/audio/ directory with .wav files\n"
            "\n"
            "You can use datasets from:\n"
            "  - Mozilla Common Voice (Hindi): https://commonvoice.mozilla.org/\n"
            "  - Google FLEURS: https://huggingface.co/datasets/google/fleurs\n"
            "  - IndicVoices: https://ai4bharat.iitm.ac.in/indicvoices/\n",
            metadata_path, data_dir, data_dir,
        )
        return None

    # Read metadata
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = audio_dir / row["file_name"]
            if audio_path.exists():
                entries.append({
                    "audio": str(audio_path),
                    "transcription": row["transcription"],
                    "language": row.get("language", "hi"),
                })
            else:
                logger.warning("Audio file not found: %s", audio_path)

    if not entries:
        logger.error("No valid audio entries found.")
        return None

    dataset = Dataset.from_list(entries)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    logger.info("Loaded %d audio samples", len(dataset))
    return dataset


def setup_model(model_name: str):
    """Load the Whisper model, processor, and feature extractor."""
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    logger.info("Loading Whisper model: %s", model_name)

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Enable gradient checkpointing for memory efficiency
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Set forced decoder IDs for Hindi
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model loaded — %d trainable parameters", trainable)

    return model, processor


def prepare_features(batch, processor):
    """
    Prepare audio features and tokenize transcriptions.

    Whisper expects:
    - input_features: log-Mel spectrogram of the audio
    - labels: tokenized transcription
    """
    audio = batch["audio"]

    # Compute log-Mel spectrogram
    input_features = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="np",
    ).input_features[0]

    # Tokenize the transcription
    labels = processor.tokenizer(batch["transcription"]).input_ids

    batch["input_features"] = input_features
    batch["labels"] = labels

    return batch


def train(
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_dir: Path,
):
    """Run the full Whisper fine-tuning pipeline."""
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
    import torch

    # Load data
    dataset = load_audio_dataset(DATA_DIR)
    if dataset is None:
        logger.error("Cannot train without data. See instructions above.")
        return

    # Load model
    model, processor = setup_model(model_name)

    # Prepare features
    logger.info("Preparing audio features...")
    dataset = dataset.map(
        lambda x: prepare_features(x, processor),
        remove_columns=["audio", "transcription", "language"],
    )

    # Split into train/eval (90/10)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    # Data collator for padding
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class WhisperDataCollator:
        processor: Any

        def __call__(self, features):
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )

            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )

            # Replace padding with -100 so loss ignores padded tokens
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # Remove BOS token if present (Whisper adds it during generation)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=10,
        report_to="none",
        seed=42,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=WhisperDataCollator(processor=processor),
        processing_class=processor,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info("Saving model to %s", output_dir)
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    logger.info("✅ STT model training complete!")


def evaluate(model_path: Path):
    """Evaluate the fine-tuned model on test samples."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    logger.info("Loading model from %s for evaluation", model_path)
    model = WhisperForConditionalGeneration.from_pretrained(str(model_path))
    processor = WhisperProcessor.from_pretrained(str(model_path))

    dataset = load_audio_dataset(DATA_DIR)
    if dataset is None:
        logger.error("No evaluation data found.")
        return

    # Take a few samples
    samples = dataset.select(range(min(5, len(dataset))))

    logger.info("\n📊 Evaluation Results:")
    for sample in samples:
        audio = sample["audio"]
        input_features = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).input_features

        predicted_ids = model.generate(input_features)
        predicted_text = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        logger.info("  Reference:  %s", sample["transcription"])
        logger.info("  Predicted:  %s", predicted_text)
        logger.info("  ---")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Sarathi STT model")
    parser.add_argument("--model", default="openai/whisper-small",
                        help="Base Whisper model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args.output_dir)
    else:
        train(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
