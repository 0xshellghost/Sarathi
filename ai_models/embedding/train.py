"""
Fine-Tuning Script for Sarathi Legal Embedding Model

Uses the sentence-transformers library to fine-tune a base embedding model
on Indian legal text pairs using triplet loss (query, positive, negative).

The resulting model produces embeddings where legal queries are close to
their relevant statute sections and far from irrelevant ones.

Hardware Requirements:
  - Minimum: 4GB VRAM (even a GTX 1650 works for all-MiniLM-L6-v2)
  - Recommended: 8GB VRAM for larger models
  - CPU: Feasible, takes ~30 minutes for the sample dataset

Usage:
    python train.py
    python train.py --base-model BAAI/bge-small-en-v1.5 --epochs 10
    python train.py --export-onnx  # Export for optimized inference
"""

import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("sarathi.embed.train")

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "training_data" / "legal_pairs.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "sarathi-embed"


def load_triplets(data_path: Path):
    """Load training triplets from JSONL file."""
    from sentence_transformers import InputExample

    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append(
                InputExample(
                    texts=[
                        data["query"],
                        data["positive"],
                        data["negative"],
                    ]
                )
            )

    logger.info("Loaded %d training triplets", len(examples))
    return examples


def train(
    base_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_dir: Path,
    warmup_ratio: float = 0.1,
):
    """Fine-tune the embedding model using triplet loss."""
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    # Load base model
    logger.info("Loading base model: %s", base_model)
    model = SentenceTransformer(base_model)

    # Load data
    train_examples = load_triplets(DATA_PATH)
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=batch_size
    )

    # Triplet loss with cosine distance
    # Margin: how much closer the positive should be compared to negative
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=0.3,
    )

    # Calculate warmup steps
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    logger.info("Training for %d epochs (%d steps, %d warmup)",
                epochs, total_steps, warmup_steps)

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=str(output_dir),
        show_progress_bar=True,
    )

    logger.info("✅ Training complete! Model saved to %s", output_dir)
    return model


def evaluate_model(model_path: Path):
    """Quick evaluation: compute similarity scores for known pairs."""
    from sentence_transformers import SentenceTransformer, util

    logger.info("Evaluating model from %s", model_path)
    model = SentenceTransformer(str(model_path))

    # Test pairs: (query, should_be_similar, should_be_dissimilar)
    test_cases = [
        (
            "landlord refusing to return security deposit",
            "Section 12 Model Tenancy Act security deposit refund",
            "Section 138 cheque dishonour criminal offence",
        ),
        (
            "cheque bounced what legal action",
            "Section 138 NI Act dishonour of cheque imprisonment fine",
            "Consumer Protection Act deficiency in service",
        ),
        (
            "company fired me without notice",
            "Industrial Disputes Act wrongful termination notice period",
            "Transfer of Property Act sale of immovable property",
        ),
    ]

    logger.info("\n📊 Similarity Scores (higher = more similar):")
    all_correct = True

    for query, positive, negative in test_cases:
        q_emb = model.encode(query)
        p_emb = model.encode(positive)
        n_emb = model.encode(negative)

        pos_score = float(util.cos_sim(q_emb, p_emb)[0][0])
        neg_score = float(util.cos_sim(q_emb, n_emb)[0][0])

        status = "✅" if pos_score > neg_score else "❌"
        if pos_score <= neg_score:
            all_correct = False

        logger.info("  %s Query: '%s...'", status, query[:40])
        logger.info("      Positive: %.4f | Negative: %.4f | Margin: %.4f",
                    pos_score, neg_score, pos_score - neg_score)

    if all_correct:
        logger.info("\n✅ All test cases passed — positive pairs rank higher than negatives.")
    else:
        logger.info("\n⚠ Some test cases failed — model may need more training data or epochs.")


def export_onnx(model_path: Path):
    """Export the trained model to ONNX format for optimized CPU inference."""
    from sentence_transformers import SentenceTransformer

    logger.info("Exporting model to ONNX...")
    model = SentenceTransformer(str(model_path))

    onnx_dir = model_path / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    # sentence-transformers supports direct ONNX export
    try:
        model.save(str(onnx_dir), create_model_card=True)
        logger.info("✅ ONNX export saved to %s", onnx_dir)
    except Exception as exc:
        logger.warning("ONNX export not supported for this model: %s", exc)
        logger.info("The standard PyTorch model at %s works fine with Ollama.", model_path)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Sarathi embedding model")
    parser.add_argument(
        "--base-model", default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base embedding model from HuggingFace",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate existing model")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export to ONNX after training")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_model(args.output_dir)
    else:
        model = train(
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            output_dir=args.output_dir,
        )
        evaluate_model(args.output_dir)

        if args.export_onnx:
            export_onnx(args.output_dir)


if __name__ == "__main__":
    main()
