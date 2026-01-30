"""Export training data to HuggingFace datasets format.

This module combines template-generated and LLM-augmented samples,
performs train/validation split, and exports to HuggingFace format.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def load_jsonl(file_path: Path) -> list[dict]:
    """Load samples from a JSONL file."""
    samples = []
    if not file_path.exists():
        return samples

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    return samples


def get_source_file_hash(sample: dict) -> str:
    """Get a hash of the source file(s) for split stratification."""
    if "metadata" in sample and "source_file" in sample["metadata"]:
        source = sample["metadata"]["source_file"]
    elif "source_file" in sample:
        source = sample["source_file"]
    elif "source_files" in sample:
        source = ",".join(sorted(sample["source_files"]))
    else:
        source = json.dumps(sample)

    return hashlib.md5(source.encode()).hexdigest()


def convert_to_chat_format(sample: dict, system_prompt: str) -> dict:
    """Convert a sample to chat format if not already."""
    if "messages" in sample:
        return sample

    # Handle augmented samples
    if "reasoning" in sample and "answer" in sample:
        question = sample.get("question", "")
        reasoning = sample.get("reasoning", "")
        answer = sample.get("answer", "")

        full_response = f"<pensamiento>\n{reasoning}\n</pensamiento>\n\n{answer}"

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": full_response},
            ],
            "metadata": {
                "source_files": sample.get("source_files", []),
                "synthesis_type": sample.get("synthesis_type", ""),
            },
        }

    return sample


def split_by_source(
    samples: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split samples ensuring no source file overlap between train/val.

    This prevents data leakage where the model sees related content
    in both training and validation.
    """
    random.seed(seed)

    # Group samples by source file hash
    by_source = {}
    for sample in samples:
        source_hash = get_source_file_hash(sample)
        if source_hash not in by_source:
            by_source[source_hash] = []
        by_source[source_hash].append(sample)

    # Shuffle source files
    source_hashes = list(by_source.keys())
    random.shuffle(source_hashes)

    # Calculate split point
    total_samples = len(samples)
    target_val = int(total_samples * val_ratio)

    # Assign sources to train/val
    train_samples = []
    val_samples = []
    val_count = 0

    for source_hash in source_hashes:
        source_samples = by_source[source_hash]
        if val_count < target_val:
            val_samples.extend(source_samples)
            val_count += len(source_samples)
        else:
            train_samples.extend(source_samples)

    # Shuffle within splits
    random.shuffle(train_samples)
    random.shuffle(val_samples)

    return train_samples, val_samples


def export_to_huggingface(
    train_samples: list[dict],
    val_samples: list[dict],
    output_dir: Path,
) -> None:
    """Export samples to HuggingFace datasets format.

    Creates a directory structure compatible with datasets.load_dataset():
        output_dir/
            train.jsonl
            validation.jsonl
            dataset_info.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write train split
    train_path = output_dir / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            # Only include messages for training
            train_sample = {"messages": sample["messages"]}
            f.write(json.dumps(train_sample, ensure_ascii=False) + "\n")

    # Write validation split
    val_path = output_dir / "validation.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for sample in val_samples:
            train_sample = {"messages": sample["messages"]}
            f.write(json.dumps(train_sample, ensure_ascii=False) + "\n")

    # Write dataset info
    info = {
        "description": "Herbolaria training dataset for traditional Mexican medicine QA",
        "features": {
            "messages": {
                "feature": {
                    "role": {"dtype": "string"},
                    "content": {"dtype": "string"},
                },
                "length": -1,
                "_type": "Sequence",
            },
        },
        "splits": {
            "train": {"num_examples": len(train_samples)},
            "validation": {"num_examples": len(val_samples)},
        },
        "version": "1.0.0",
    }

    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"Exported to {output_dir}:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Validation: {len(val_samples)} samples")


def export_for_axolotl(
    train_samples: list[dict],
    val_samples: list[dict],
    output_dir: Path,
) -> None:
    """Export samples in Axolotl-compatible format.

    Axolotl expects a specific chat format for SFT training.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def convert_for_axolotl(sample: dict) -> dict:
        """Convert to Axolotl format."""
        messages = sample["messages"]
        return {"conversations": messages}

    # Write train split
    train_path = output_dir / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            axolotl_sample = convert_for_axolotl(sample)
            f.write(json.dumps(axolotl_sample, ensure_ascii=False) + "\n")

    # Write validation split
    val_path = output_dir / "val.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for sample in val_samples:
            axolotl_sample = convert_for_axolotl(sample)
            f.write(json.dumps(axolotl_sample, ensure_ascii=False) + "\n")

    print(f"Exported Axolotl format to {output_dir}:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Validation: {len(val_samples)} samples")


# Default system prompt
SYSTEM_PROMPT = """Eres un asistente experto en medicina tradicional mexicana. Tu conocimiento abarca:

- Plantas medicinales mexicanas y sus usos terapéuticos
- Síndromes de filiación cultural (susto, mal de ojo, empacho, etc.)
- Prácticas y rituales de curación tradicionales
- Conocimientos de los pueblos indígenas de México
- Historia de la medicina tradicional desde la época prehispánica

Respondes en español con precisión y respeto cultural. Cuando es relevante, mencionas:
- Nombres en lenguas indígenas (náhuatl, maya, etc.)
- Usos regionales específicos
- Precauciones y contraindicaciones
- Fuentes históricas cuando están disponibles

Siempre aclaras que la información es con fines educativos y culturales, no como consejo médico."""


def main():
    """CLI entry point for exporting datasets."""
    parser = argparse.ArgumentParser(
        description="Export training data to HuggingFace/Axolotl format"
    )
    parser.add_argument(
        "--template-samples",
        type=Path,
        default=Path(__file__).parent.parent / "training_samples.jsonl",
        help="Template-generated samples file",
    )
    parser.add_argument(
        "--augmented-samples",
        type=Path,
        default=Path(__file__).parent.parent / "augmented_samples.jsonl",
        help="LLM-augmented samples file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent.parent / "data" / "herbolaria_training",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--format",
        choices=["huggingface", "axolotl", "both"],
        default="both",
        help="Export format",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum total samples (for testing)",
    )
    args = parser.parse_args()

    # Load samples
    print("Loading samples...")
    template_samples = load_jsonl(args.template_samples)
    augmented_samples = load_jsonl(args.augmented_samples)

    print(f"  Template samples: {len(template_samples)}")
    print(f"  Augmented samples: {len(augmented_samples)}")

    # Convert augmented samples to chat format
    augmented_samples = [
        convert_to_chat_format(s, SYSTEM_PROMPT) for s in augmented_samples
    ]

    # Combine all samples
    all_samples = template_samples + augmented_samples
    print(f"  Total samples: {len(all_samples)}")

    # Limit if requested
    if args.max_samples and len(all_samples) > args.max_samples:
        random.seed(args.seed)
        random.shuffle(all_samples)
        all_samples = all_samples[: args.max_samples]
        print(f"  Limited to: {len(all_samples)} samples")

    # Split
    print("\nSplitting dataset...")
    train_samples, val_samples = split_by_source(
        all_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Export
    if args.format in ("huggingface", "both"):
        hf_output = args.output
        export_to_huggingface(train_samples, val_samples, hf_output)

    if args.format in ("axolotl", "both"):
        axolotl_output = args.output.parent / f"{args.output.name}_axolotl"
        export_for_axolotl(train_samples, val_samples, axolotl_output)

    # Print sample for verification
    print("\n=== Sample Training Example ===")
    if train_samples:
        sample = train_samples[0]
        for msg in sample["messages"]:
            role = msg["role"].upper()
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            print(f"\n[{role}]")
            print(content)


if __name__ == "__main__":
    main()
