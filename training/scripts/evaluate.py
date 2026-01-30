"""Evaluation script for the finetuned herbolaria model.

This script evaluates the model on:
1. Held-out validation set (perplexity, loss)
2. Domain-specific benchmark questions
3. Comparison with base model

Usage:
    python training/scripts/evaluate.py \
        --model_path models/herbolaria-dasd-4b-merged \
        --eval_dataset data/herbolaria_training/validation.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Domain-specific benchmark questions
BENCHMARK_QUESTIONS = [
    {
        "question": "¿Cuáles son los usos medicinales de la sábila?",
        "expected_topics": ["diabetes", "quemaduras", "piel", "cicatrizante", "Aloe vera"],
        "category": "plantas_general",
    },
    {
        "question": "¿Qué es el mal de ojo y cómo se trata?",
        "expected_topics": ["niños", "limpia", "huevo", "envidia", "mirada"],
        "category": "sindrome_cultural",
    },
    {
        "question": "¿Cuáles son los compuestos químicos del estafiate?",
        "expected_topics": ["artemisia", "aceite esencial", "terpenos"],
        "category": "quimica",
    },
    {
        "question": "¿Cómo es la medicina tradicional de los nahuas?",
        "expected_topics": ["nahua", "curandero", "temazcal", "puebla", "veracruz"],
        "category": "pueblos",
    },
    {
        "question": "¿Qué precauciones hay que tomar al usar la ruda?",
        "expected_topics": ["toxicidad", "embarazo", "aborto", "dosis"],
        "category": "toxicidad",
    },
    {
        "question": "¿Qué plantas se usan para tratar la diabetes en la medicina tradicional mexicana?",
        "expected_topics": ["nopal", "sábila", "wereke", "tronadora", "hipoglucemiante"],
        "category": "tratamiento",
    },
    {
        "question": "¿Qué es el susto y cuáles son sus síntomas?",
        "expected_topics": ["espanto", "pérdida del alma", "fiebre", "debilidad", "limpia"],
        "category": "sindrome_cultural",
    },
    {
        "question": "¿Cuál es la historia del uso medicinal del cuachalalate?",
        "expected_topics": ["historia", "prehispánico", "colonial", "úlceras", "gastritis"],
        "category": "historia",
    },
]


def load_model(model_path: str, load_in_4bit: bool = False):
    """Load model and tokenizer."""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    return model, tokenizer


def calculate_perplexity(
    model,
    tokenizer,
    eval_file: Path,
    max_samples: int = 100,
) -> dict[str, float]:
    """Calculate perplexity on the evaluation dataset."""
    print("Calculating perplexity...")

    samples = []
    with open(eval_file) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= max_samples:
                break

    total_loss = 0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for sample in tqdm(samples, desc="Evaluating"):
            messages = sample.get("messages", [])

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(model.device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "num_samples": len(samples),
        "num_tokens": total_tokens,
    }


def evaluate_benchmark(
    model,
    tokenizer,
    questions: list[dict],
    max_new_tokens: int = 512,
) -> list[dict[str, Any]]:
    """Evaluate on benchmark questions."""
    print("Evaluating benchmark questions...")

    system_prompt = """Eres un asistente experto en medicina tradicional mexicana. Respondes en español con precisión y respeto cultural."""

    results = []

    for q in tqdm(questions, desc="Benchmark"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q["question"]},
        ]

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Check for expected topics
        response_lower = response.lower()
        topics_found = [
            topic for topic in q["expected_topics"]
            if topic.lower() in response_lower
        ]
        topic_coverage = len(topics_found) / len(q["expected_topics"])

        results.append({
            "question": q["question"],
            "response": response,
            "category": q["category"],
            "expected_topics": q["expected_topics"],
            "topics_found": topics_found,
            "topic_coverage": topic_coverage,
        })

    return results


def compare_models(
    finetuned_path: str,
    base_model_name: str,
    questions: list[dict],
    max_new_tokens: int = 512,
) -> list[dict]:
    """Compare finetuned model with base model."""
    print("Loading finetuned model...")
    ft_model, ft_tokenizer = load_model(finetuned_path)

    print("Loading base model...")
    base_model, base_tokenizer = load_model(base_model_name)

    print("Comparing models...")

    system_prompt = """Eres un asistente experto en medicina tradicional mexicana."""

    results = []

    for q in tqdm(questions[:5], desc="Comparing"):  # Limit for speed
        # Finetuned response
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q["question"]},
        ]

        ft_input = ft_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ft_inputs = ft_tokenizer(ft_input, return_tensors="pt").to(ft_model.device)

        with torch.no_grad():
            ft_outputs = ft_model.generate(
                **ft_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=ft_tokenizer.pad_token_id,
            )

        ft_response = ft_tokenizer.decode(
            ft_outputs[0][ft_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Base model response
        base_input = base_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        base_inputs = base_tokenizer(base_input, return_tensors="pt").to(base_model.device)

        with torch.no_grad():
            base_outputs = base_model.generate(
                **base_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=base_tokenizer.pad_token_id,
            )

        base_response = base_tokenizer.decode(
            base_outputs[0][base_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Score both responses
        ft_topics = [
            t for t in q["expected_topics"]
            if t.lower() in ft_response.lower()
        ]
        base_topics = [
            t for t in q["expected_topics"]
            if t.lower() in base_response.lower()
        ]

        results.append({
            "question": q["question"],
            "finetuned_response": ft_response,
            "base_response": base_response,
            "finetuned_topic_coverage": len(ft_topics) / len(q["expected_topics"]),
            "base_topic_coverage": len(base_topics) / len(q["expected_topics"]),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate the finetuned model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the finetuned model",
    )
    parser.add_argument(
        "--eval_dataset",
        type=Path,
        default=Path("data/herbolaria_training/validation.jsonl"),
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/evaluation_results.json"),
        help="Output file for results",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Alibaba-Apsara/DASD-4B-Thinking",
        help="Base model for comparison",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with base model",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    args = parser.parse_args()

    results = {}

    # Load model
    model, tokenizer = load_model(args.model_path, args.load_in_4bit)

    # Calculate perplexity
    if args.eval_dataset.exists():
        perplexity_results = calculate_perplexity(model, tokenizer, args.eval_dataset)
        results["perplexity"] = perplexity_results
        print(f"\nPerplexity: {perplexity_results['perplexity']:.2f}")
        print(f"Loss: {perplexity_results['loss']:.4f}")

    # Benchmark evaluation
    benchmark_results = evaluate_benchmark(model, tokenizer, BENCHMARK_QUESTIONS)
    results["benchmark"] = benchmark_results

    # Calculate average topic coverage
    avg_coverage = sum(r["topic_coverage"] for r in benchmark_results) / len(benchmark_results)
    results["avg_topic_coverage"] = avg_coverage
    print(f"\nAverage topic coverage: {avg_coverage:.2%}")

    # Print benchmark results
    print("\nBenchmark Results:")
    for r in benchmark_results:
        print(f"  {r['category']}: {r['topic_coverage']:.0%} coverage")
        print(f"    Q: {r['question'][:50]}...")
        print(f"    Found: {r['topics_found']}")

    # Compare with base model if requested
    if args.compare:
        comparison = compare_models(
            args.model_path,
            args.base_model,
            BENCHMARK_QUESTIONS,
        )
        results["comparison"] = comparison

        print("\nModel Comparison:")
        for r in comparison:
            print(f"  Q: {r['question'][:50]}...")
            print(f"    Finetuned: {r['finetuned_topic_coverage']:.0%}")
            print(f"    Base: {r['base_topic_coverage']:.0%}")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
