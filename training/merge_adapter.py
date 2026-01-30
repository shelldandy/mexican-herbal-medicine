"""Merge LoRA adapter weights with base model.

After training with QLoRA, the adapter weights need to be merged with the
base model for efficient inference. This script performs that merge and
optionally quantizes the result.

Usage:
    python training/merge_adapter.py \
        --adapter_path "models/herbolaria-dasd-4b-lora/final" \
        --output_path "models/herbolaria-dasd-4b-merged"
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora_adapter(
    base_model_name: str,
    adapter_path: Path,
    output_path: Path,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
) -> None:
    """Merge LoRA adapter weights with base model.

    Args:
        base_model_name: Name or path of the base model.
        adapter_path: Path to the LoRA adapter weights.
        output_path: Path to save the merged model.
        torch_dtype: Data type for model weights.
        device_map: Device mapping strategy.
        push_to_hub: Whether to push to HuggingFace Hub.
        hub_repo_id: Repository ID for Hub push.
    """
    # Map string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=dtype,
    )

    print("Merging adapter weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(output_path)

    # Push to Hub if requested
    if push_to_hub and hub_repo_id:
        print(f"Pushing to HuggingFace Hub: {hub_repo_id}")
        model.push_to_hub(hub_repo_id)
        tokenizer.push_to_hub(hub_repo_id)

    print("Done!")
    print(f"\nMerged model saved to: {output_path}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")


def export_to_gguf(
    model_path: Path,
    output_path: Path,
    quantization: str = "q4_k_m",
) -> None:
    """Export model to GGUF format for llama.cpp.

    Requires llama.cpp's convert script to be available.

    Args:
        model_path: Path to the merged model.
        output_path: Path for the GGUF output.
        quantization: Quantization type (q4_k_m, q8_0, etc.)
    """
    import subprocess

    print(f"Converting to GGUF format with {quantization} quantization...")

    # First convert to GGUF
    gguf_path = output_path.with_suffix(".gguf")

    # This requires llama.cpp to be installed
    # The convert script location may vary
    convert_cmd = [
        "python",
        "-m",
        "llama_cpp.convert",
        str(model_path),
        "--outfile",
        str(gguf_path),
        "--outtype",
        quantization,
    ]

    try:
        subprocess.run(convert_cmd, check=True)
        print(f"GGUF model saved to: {gguf_path}")
    except FileNotFoundError:
        print("Note: llama-cpp-python not found. Install with:")
        print("  pip install llama-cpp-python")
        print("\nAlternatively, use llama.cpp directly:")
        print(f"  python convert-hf-to-gguf.py {model_path} --outfile {gguf_path}")


def main():
    """Main entry point for merging adapters."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter weights with base model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Alibaba-Apsara/DASD-4B-Thinking",
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter_path",
        type=Path,
        required=True,
        help="Path to the LoRA adapter",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Output path for merged model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Model weight data type",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        help="HuggingFace Hub repository ID",
    )
    parser.add_argument(
        "--export_gguf",
        action="store_true",
        help="Also export to GGUF format",
    )
    parser.add_argument(
        "--gguf_quantization",
        type=str,
        default="q4_k_m",
        help="GGUF quantization type",
    )
    args = parser.parse_args()

    # Merge adapter
    merge_lora_adapter(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        torch_dtype=args.dtype,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
    )

    # Export to GGUF if requested
    if args.export_gguf:
        gguf_output = args.output_path.parent / f"{args.output_path.name}.gguf"
        export_to_gguf(
            model_path=args.output_path,
            output_path=gguf_output,
            quantization=args.gguf_quantization,
        )


if __name__ == "__main__":
    main()
