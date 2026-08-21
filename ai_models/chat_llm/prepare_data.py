"""
Training Data Preparation for Sarathi Legal Chat LLM

Converts the raw JSONL training conversations into the format required
by the fine-tuning framework (unsloth/HuggingFace). Handles:
  - Splitting into train/validation sets
  - Statistics reporting
  - Format validation
  - Augmentation (optional paraphrasing of user queries)

Usage:
    python prepare_data.py
    python prepare_data.py --augment   # Enable query paraphrasing
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter


SCRIPT_DIR = Path(__file__).parent
RAW_DATA = SCRIPT_DIR / "training_data" / "legal_conversations.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "training_data" / "processed"


def load_conversations(path: Path) -> list[dict]:
    """Load JSONL training data and validate structure."""
    conversations = []
    errors = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                messages = entry.get("messages", [])

                # Validate structure: must have system, user, assistant
                roles = [m["role"] for m in messages]
                if "system" not in roles or "user" not in roles or "assistant" not in roles:
                    errors.append(f"Line {line_num}: Missing required roles (need system, user, assistant)")
                    continue

                # Validate no empty content
                for msg in messages:
                    if not msg.get("content", "").strip():
                        errors.append(f"Line {line_num}: Empty content for role '{msg['role']}'")
                        continue

                conversations.append(entry)

            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON — {e}")

    if errors:
        print(f"\n⚠ {len(errors)} validation errors:")
        for err in errors[:10]:
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return conversations


def classify_task(entry: dict) -> str:
    """Determine the task type from the system prompt."""
    system_content = ""
    for msg in entry["messages"]:
        if msg["role"] == "system":
            system_content = msg["content"].lower()
            break

    if "classify" in system_content or "domain" in system_content:
        return "intent_classification"
    elif "extract" in system_content or "entities" in system_content:
        return "entity_extraction"
    elif "explain" in system_content or "rights" in system_content:
        return "legal_explanation"
    else:
        return "unknown"


def print_statistics(conversations: list[dict]) -> None:
    """Print dataset statistics."""
    task_counts = Counter(classify_task(c) for c in conversations)

    # Count domains from intent classification and entity extraction tasks
    domain_counts = Counter()
    for conv in conversations:
        assistant_msg = ""
        for msg in conv["messages"]:
            if msg["role"] == "assistant":
                assistant_msg = msg["content"]
                break
        try:
            data = json.loads(assistant_msg)
            if "domain" in data:
                domain_counts[data["domain"]] += 1
        except (json.JSONDecodeError, TypeError):
            pass

    print("\n" + "=" * 50)
    print("📊 Dataset Statistics")
    print("=" * 50)
    print(f"Total conversations: {len(conversations)}")
    print(f"\nBy task type:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")
    print(f"\nBy legal domain (where applicable):")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")
    print()


def split_dataset(
    conversations: list[dict],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split into train and validation sets, stratified by task type."""
    random.seed(seed)

    # Group by task type for stratified splitting
    by_task: dict[str, list[dict]] = {}
    for conv in conversations:
        task = classify_task(conv)
        by_task.setdefault(task, []).append(conv)

    train, val = [], []
    for task, items in by_task.items():
        random.shuffle(items)
        split_idx = max(1, int(len(items) * val_ratio))
        val.extend(items[:split_idx])
        train.extend(items[split_idx:])

    random.shuffle(train)
    random.shuffle(val)

    return train, val


def convert_to_chat_format(conversations: list[dict]) -> list[dict]:
    """
    Convert to the HuggingFace chat format used by unsloth/TRL.

    Output format per entry:
    {
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }
    """
    formatted = []
    for conv in conversations:
        entry = {"conversations": []}
        for msg in conv["messages"]:
            role_map = {"system": "system", "user": "human", "assistant": "gpt"}
            entry["conversations"].append({
                "from": role_map.get(msg["role"], msg["role"]),
                "value": msg["content"],
            })
        formatted.append(entry)
    return formatted


def save_jsonl(data: list[dict], path: Path) -> None:
    """Write data as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} entries to {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Sarathi LLM training data")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("Loading raw training data...")
    conversations = load_conversations(RAW_DATA)

    if not conversations:
        print("No valid conversations found. Check the training data file.")
        return

    print_statistics(conversations)

    # Split
    train_data, val_data = split_dataset(conversations, args.val_ratio, args.seed)
    print(f"Split: {len(train_data)} train, {len(val_data)} validation")

    # Convert to chat format
    train_formatted = convert_to_chat_format(train_data)
    val_formatted = convert_to_chat_format(val_data)

    # Save
    print("\nSaving processed datasets...")
    save_jsonl(train_formatted, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val_formatted, OUTPUT_DIR / "val.jsonl")

    # Also save the raw splits for reference
    save_jsonl(train_data, OUTPUT_DIR / "train_raw.jsonl")
    save_jsonl(val_data, OUTPUT_DIR / "val_raw.jsonl")

    print("\n✅ Data preparation complete!")
    print(f"   Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
