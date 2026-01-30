"""Training script for finetuning DASD-4B-Thinking on herbolaria dataset.

This script implements QLoRA training with the following features:
- 4-bit quantization for memory efficiency
- LoRA adapters for parameter-efficient finetuning
- Gradient checkpointing for reduced memory usage
- Wandb/TensorBoard logging for monitoring

Usage:
    python training/train.py --config training/configs/qlora_config.yaml

    # Or with command-line overrides:
    python training/train.py \
        --model_name "Alibaba-Apsara/DASD-4B-Thinking" \
        --dataset_path "data/herbolaria_training" \
        --output_dir "models/herbolaria-dasd-4b-lora"
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_configs(base_config: dict, overrides: dict) -> dict:
    """Merge command-line overrides into base config."""
    config = base_config.copy()

    # Apply overrides
    for key, value in overrides.items():
        if value is not None:
            # Handle nested keys like model.name -> config['model']['name']
            if "." in key:
                parts = key.split(".")
                d = config
                for part in parts[:-1]:
                    if part not in d:
                        d[part] = {}
                    d = d[part]
                d[parts[-1]] = value
            else:
                config[key] = value

    return config


def create_quantization_config(config: dict) -> BitsAndBytesConfig:
    """Create BitsAndBytes quantization configuration."""
    quant_config = config.get("quantization", {})

    # Map string dtype to torch dtype
    compute_dtype = quant_config.get("bnb_4bit_compute_dtype", "bfloat16")
    if compute_dtype == "bfloat16":
        compute_dtype = torch.bfloat16
    elif compute_dtype == "float16":
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    return BitsAndBytesConfig(
        load_in_4bit=quant_config.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
    )


def create_lora_config(config: dict) -> LoraConfig:
    """Create LoRA configuration."""
    lora_config = config.get("lora", {})

    return LoraConfig(
        r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("lora_alpha", 128),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
        target_modules=lora_config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )


def create_training_arguments(config: dict) -> TrainingArguments:
    """Create training arguments from config."""
    training_config = config.get("training", {})

    return TrainingArguments(
        output_dir=training_config.get("output_dir", "models/output"),
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        learning_rate=training_config.get("learning_rate", 2e-4),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.03),
        weight_decay=training_config.get("weight_decay", 0.01),
        optim=training_config.get("optim", "paged_adamw_8bit"),
        bf16=training_config.get("bf16", True),
        tf32=training_config.get("tf32", True),
        logging_steps=training_config.get("logging_steps", 10),
        logging_first_step=training_config.get("logging_first_step", True),
        eval_strategy=training_config.get("eval_strategy", "steps"),
        eval_steps=training_config.get("eval_steps", 100),
        save_strategy=training_config.get("save_strategy", "steps"),
        save_steps=training_config.get("save_steps", 100),
        save_total_limit=training_config.get("save_total_limit", 3),
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_config.get("greater_is_better", False),
        seed=training_config.get("seed", 42),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        remove_unused_columns=training_config.get("remove_unused_columns", False),
        report_to=training_config.get("report_to", "wandb"),
        gradient_checkpointing_kwargs={"use_reentrant": False}
        if training_config.get("gradient_checkpointing", True)
        else None,
    )


def format_chat_messages(example: dict, tokenizer) -> dict:
    """Format messages using the tokenizer's chat template."""
    messages = example["messages"]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train DASD-4B on herbolaria dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "qlora_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--model_name", type=str, help="Model name or path")
    parser.add_argument("--dataset_path", type=str, help="Dataset path")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Per-device batch size")
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )
    args = parser.parse_args()

    # Load base config
    if args.config.exists():
        config = load_config(args.config)
    else:
        config = {}

    # Apply command-line overrides
    overrides = {
        "model.name": args.model_name,
        "dataset.path": args.dataset_path,
        "training.output_dir": args.output_dir,
        "training.num_train_epochs": args.num_train_epochs,
        "training.learning_rate": args.learning_rate,
        "training.per_device_train_batch_size": args.batch_size,
    }
    config = merge_configs(config, overrides)

    if args.no_wandb:
        config["training"]["report_to"] = "none"

    # Set up wandb if enabled
    if config.get("training", {}).get("report_to") == "wandb":
        wandb_config = config.get("wandb", {})
        os.environ.setdefault("WANDB_PROJECT", wandb_config.get("project", "herbolaria-finetune"))
        os.environ.setdefault("WANDB_RUN_NAME", wandb_config.get("name", "dasd-4b-qlora"))

    # Get model name
    model_name = config.get("model", {}).get("name", "Alibaba-Apsara/DASD-4B-Thinking")
    print(f"Loading model: {model_name}")

    # Create quantization config
    bnb_config = create_quantization_config(config)
    print("Quantization config created (4-bit NF4)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config.get("model", {}).get("trust_remote_code", True),
    )

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Tokenizer loaded: {len(tokenizer)} tokens")

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config.get("model", {}).get("trust_remote_code", True),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
        if config.get("memory", {}).get("use_flash_attention_2", True)
        else "eager",
    )

    print(f"Model loaded: {model.num_parameters():,} parameters")

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Create LoRA config and apply
    lora_config = create_lora_config(config)
    model = get_peft_model(model, lora_config)

    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

    # Load dataset
    dataset_path = config.get("dataset", {}).get("path", "data/herbolaria_training")
    print(f"Loading dataset from: {dataset_path}")

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(Path(dataset_path) / "train.jsonl"),
            "validation": str(Path(dataset_path) / "validation.jsonl"),
        },
    )

    print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} validation")

    # Format dataset for training
    def formatting_func(example):
        return format_chat_messages(example, tokenizer)["text"]

    # Create training arguments
    training_args = create_training_arguments(config)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        formatting_func=formatting_func,
        max_seq_length=config.get("sft", {}).get("max_seq_length", 2048),
        packing=config.get("sft", {}).get("packing", False),
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    final_output_dir = Path(training_args.output_dir) / "final"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    print(f"Model saved to: {final_output_dir}")

    # Print final metrics
    metrics = trainer.evaluate()
    print("\nFinal evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
