# Finetuning DASD-4B-Thinking on Herbolaria Dataset

This directory contains the infrastructure for finetuning a 4B parameter model on the traditional Mexican medicine dataset.

## Overview

- **Base Model**: [DASD-4B-Thinking](https://huggingface.co/Alibaba-Apsara/DASD-4B-Thinking) (Qwen3-4B based)
- **Data**: 3,720 markdown files with rich structured sections
- **Target**: 1,500-2,000 high-quality instruction-response pairs with chain-of-thought reasoning
- **Method**: QLoRA (4-bit quantization + LoRA adapters)

## Directory Structure

```
training/
├── data_preparation/
│   ├── parse_documents.py     # Parse markdown files, extract sections
│   ├── generate_qa.py         # Template-based Q&A generation
│   ├── augment_with_llm.py    # LLM-assisted generation (Claude/GPT-4)
│   └── export_dataset.py      # Export to HuggingFace format
├── configs/
│   └── qlora_config.yaml      # Training configuration
├── scripts/
│   ├── serve_model.py         # Model serving (local/vLLM/Gradio)
│   ├── local_llm.py           # RAG app integration
│   └── evaluate.py            # Model evaluation
├── train.py                   # Main training script
├── merge_adapter.py           # Merge LoRA weights with base model
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Quick Start

### 1. Install Dependencies

```bash
cd training
pip install -r requirements.txt

# Optional: Flash Attention for faster training (CUDA 11.6+ required)
pip install flash-attn --no-build-isolation
```

### 2. Generate Training Data

```bash
# Parse all documents and show statistics
python -m training.data_preparation.parse_documents --data-dir data/ --stats

# Generate Q&A pairs from templates
python -m training.data_preparation.generate_qa --data-dir data/ --output training/training_samples.jsonl

# (Optional) Generate LLM-augmented samples
export ANTHROPIC_API_KEY=your_key
python -m training.data_preparation.augment_with_llm --data-dir data/ --max-samples 100

# Export to HuggingFace format with train/val split
python -m training.data_preparation.export_dataset \
    --template-samples training/training_samples.jsonl \
    --output data/herbolaria_training
```

### 3. Train the Model

```bash
# Using config file
python training/train.py --config training/configs/qlora_config.yaml

# Or with command-line arguments
python training/train.py \
    --model_name "Alibaba-Apsara/DASD-4B-Thinking" \
    --dataset_path "data/herbolaria_training" \
    --output_dir "models/herbolaria-dasd-4b-lora" \
    --num_train_epochs 3

# Disable wandb if not configured
python training/train.py --config training/configs/qlora_config.yaml --no_wandb
```

### 4. Merge Adapter Weights

```bash
python training/merge_adapter.py \
    --adapter_path models/herbolaria-dasd-4b-lora/final \
    --output_path models/herbolaria-dasd-4b-merged
```

### 5. Evaluate

```bash
python training/scripts/evaluate.py \
    --model_path models/herbolaria-dasd-4b-merged \
    --eval_dataset data/herbolaria_training/validation.jsonl
```

### 6. Serve the Model

```bash
# Interactive local session
python training/scripts/serve_model.py --model_path models/herbolaria-dasd-4b-merged --mode local

# vLLM server (production)
python training/scripts/serve_model.py --model_path models/herbolaria-dasd-4b-merged --mode vllm --port 8000

# Gradio web interface
python training/scripts/serve_model.py --model_path models/herbolaria-dasd-4b-merged --mode gradio --port 7860
```

## Training Configuration

The default configuration in `configs/qlora_config.yaml` uses:

- **Quantization**: 4-bit NF4 with double quantization
- **LoRA**: r=64, alpha=128, targeting attention and MLP layers
- **Training**: 3 epochs, batch size 2 (effective 16 with accumulation), cosine LR schedule
- **Memory**: Flash Attention 2, gradient checkpointing

### Hardware Requirements

| Option          | VRAM | Notes                               |
| --------------- | ---- | ----------------------------------- |
| RTX 4090        | 24GB | Consumer GPU, suitable for training |
| Cloud A10G      | 24GB | ~$1-2/hr on cloud providers         |
| Cloud A100-40GB | 40GB | Faster training, ~$3-5/hr           |

### Adjusting for Lower VRAM

If you have less than 24GB VRAM, try:

```yaml
# In qlora_config.yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
sft:
  max_seq_length: 1024
```

## Integration with RAG App

To use the finetuned model with the existing Streamlit RAG app:

```python
from training.scripts.local_llm import LocalHerbolariLLM

# Initialize
llm = LocalHerbolariLLM("./models/herbolaria-dasd-4b-merged")

# Generate with RAG context
context = "La sábila (Aloe vera) es una planta con propiedades..."
response = llm.generate(
    query="¿Cómo se usa la sábila para la diabetes?",
    context=context
)

# Streaming generation
for chunk in llm.generate_stream(query, context):
    print(chunk, end="", flush=True)
```

Or use the vLLM client for production:

```python
from training.scripts.local_llm import VLLMClient

client = VLLMClient(base_url="http://localhost:8000/v1")
response = client.generate("¿Qué plantas se usan para la diabetes?")
```

## Data Format

Training samples use the chat format with chain-of-thought reasoning:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Eres un asistente experto en medicina tradicional mexicana..."
    },
    {
      "role": "user",
      "content": "¿Cuáles son los usos medicinales de la sábila?"
    },
    {
      "role": "assistant",
      "content": "<pensamiento>\nVoy a analizar la información disponible sobre Sábila (Aloe vera)...\n</pensamiento>\n\n**Sábila** (Aloe vera)\n\nLa sábila tiene múltiples usos medicinales en la medicina tradicional mexicana..."
    }
  ]
}
```

## Evaluation Metrics

The evaluation script (`scripts/evaluate.py`) measures:

1. **Perplexity**: Language modeling quality on held-out data
2. **Topic Coverage**: Whether responses include expected domain terms
3. **Model Comparison**: Side-by-side with base model (optional)

Benchmark questions cover:

- Plant medicinal uses (general, chemistry, toxicity)
- Cultural syndromes (susto, mal de ojo)
- Historical knowledge
- Indigenous medicine traditions

## Troubleshooting

### CUDA Out of Memory

- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` proportionally
- Reduce `max_seq_length` to 1024
- Enable `optimizer_offload: true` in config

### Flash Attention Installation Issues

Flash Attention requires CUDA 11.6+ and specific GPU architectures:

```bash
# Check CUDA version
nvcc --version

# Install without build isolation
pip install flash-attn --no-build-isolation

# Or disable in config
memory:
  use_flash_attention_2: false
```

### Wandb Login

```bash
wandb login
# Or set environment variable
export WANDB_API_KEY=your_key
```

## References

- [How to finetune Qwen - Complete Guide 2025](https://apatero.com/blog/how-to-finetune-qwen-complete-guide-2025)
- [Fine-tune LLMs in 2025 with Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2025)
- [DASD-4B-Thinking Model Card](https://huggingface.co/Alibaba-Apsara/DASD-4B-Thinking)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
