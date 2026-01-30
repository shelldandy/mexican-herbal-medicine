"""Augment training data using LLM-generated questions.

This module uses Claude or other LLMs to generate more complex,
reasoning-heavy questions that require synthesis across sources.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from .parse_documents import ParsedDocument, parse_all_documents

# Complex question templates that require reasoning across sections
SYNTHESIS_PROMPTS = [
    {
        "type": "comparison",
        "prompt": """Genera una pregunta que requiera comparar los usos medicinales de dos o más plantas para tratar un mismo padecimiento.

Plantas disponibles:
{plant_list}

La pregunta debe:
1. Ser específica sobre un padecimiento común
2. Requerir analizar las propiedades de múltiples plantas
3. Poder responderse con la información disponible

Formato de respuesta:
PREGUNTA: [tu pregunta]
RAZONAMIENTO: [pasos de razonamiento para responder]
RESPUESTA: [respuesta basada en las plantas]""",
    },
    {
        "type": "mechanism",
        "prompt": """Genera una pregunta sobre el mecanismo de acción o principios activos de una planta medicinal.

Información de la planta {plant_name}:
Química: {quimica}
Farmacología: {farmacologia}

La pregunta debe:
1. Conectar los compuestos químicos con los efectos observados
2. Requerir razonamiento sobre la relación estructura-actividad
3. Ser educativa sobre fitoquímica

Formato de respuesta:
PREGUNTA: [tu pregunta]
RAZONAMIENTO: [análisis de la relación química-efecto]
RESPUESTA: [explicación del mecanismo]""",
    },
    {
        "type": "cultural",
        "prompt": """Genera una pregunta sobre el contexto cultural de una práctica médica tradicional.

Término o concepto: {term}
Descripción: {description}

La pregunta debe:
1. Explorar el significado cultural del concepto
2. Relacionarlo con la cosmovisión indígena
3. Contextualizar dentro de la medicina tradicional mexicana

Formato de respuesta:
PREGUNTA: [tu pregunta]
RAZONAMIENTO: [análisis del contexto cultural]
RESPUESTA: [explicación cultural]""",
    },
    {
        "type": "historical",
        "prompt": """Genera una pregunta sobre la evolución histórica del uso de una planta medicinal.

Planta: {plant_name}
Historia: {historia}
Usos actuales (Etnobotánica): {etnobotanica}

La pregunta debe:
1. Comparar usos históricos con actuales
2. Mencionar fuentes coloniales si están disponibles
3. Analizar la continuidad o cambio de prácticas

Formato de respuesta:
PREGUNTA: [tu pregunta]
RAZONAMIENTO: [análisis histórico]
RESPUESTA: [evolución del uso]""",
    },
    {
        "type": "safety",
        "prompt": """Genera una pregunta sobre seguridad y contraindicaciones de una planta medicinal.

Planta: {plant_name}
Toxicidad: {toxicidad}
Usos tradicionales: {etnobotanica}

La pregunta debe:
1. Abordar aspectos de seguridad importantes
2. Mencionar grupos de riesgo si aplica
3. Balancear beneficios y riesgos

Formato de respuesta:
PREGUNTA: [tu pregunta]
RAZONAMIENTO: [análisis de riesgo-beneficio]
RESPUESTA: [recomendaciones de seguridad]""",
    },
]


@dataclass
class AugmentedSample:
    """An LLM-augmented training sample."""

    question: str
    reasoning: str
    answer: str
    synthesis_type: str
    source_files: list[str]


def create_llm_prompt_for_comparison(
    plantas_docs: list[ParsedDocument],
) -> tuple[str, list[str]]:
    """Create a prompt for generating comparison questions."""
    # Select 2-3 plants that share common uses
    # Group by common symptoms/uses found in etnobotanica
    common_uses = {}
    for doc in plantas_docs:
        etnobotanica = doc.get_section("Etnobotánica") or ""
        # Extract common ailments mentioned
        ailments = [
            "diabetes",
            "tos",
            "dolor de estómago",
            "inflamación",
            "heridas",
            "fiebre",
            "diarrea",
            "dolor de cabeza",
        ]
        for ailment in ailments:
            if ailment.lower() in etnobotanica.lower():
                if ailment not in common_uses:
                    common_uses[ailment] = []
                common_uses[ailment].append(doc)

    # Find an ailment with multiple plants
    for ailment, docs in common_uses.items():
        if len(docs) >= 2:
            selected = docs[:3]
            plant_list = "\n".join(
                [
                    f"- {d.title} ({d.botanical_name}): {(d.get_section('Etnobotánica') or '')[:300]}"
                    for d in selected
                ]
            )
            prompt = SYNTHESIS_PROMPTS[0]["prompt"].format(plant_list=plant_list)
            source_files = [d.file_path for d in selected]
            return prompt, source_files

    return "", []


def create_llm_prompt_for_mechanism(doc: ParsedDocument) -> tuple[str, list[str]]:
    """Create a prompt for generating mechanism questions."""
    quimica = doc.get_section("Química")
    farmacologia = doc.get_section("Farmacología")

    if not quimica or not farmacologia:
        return "", []

    prompt = SYNTHESIS_PROMPTS[1]["prompt"].format(
        plant_name=doc.title,
        quimica=quimica[:500],
        farmacologia=farmacologia[:500],
    )
    return prompt, [doc.file_path]


def create_llm_prompt_for_cultural(doc: ParsedDocument) -> tuple[str, list[str]]:
    """Create a prompt for generating cultural context questions."""
    if doc.doc_type != "diccionario":
        return "", []

    descripcion = doc.get_section("Descripción") or doc.raw_body

    if not descripcion or len(descripcion) < 100:
        return "", []

    prompt = SYNTHESIS_PROMPTS[2]["prompt"].format(
        term=doc.title,
        description=descripcion[:800],
    )
    return prompt, [doc.file_path]


def create_llm_prompt_for_historical(doc: ParsedDocument) -> tuple[str, list[str]]:
    """Create a prompt for generating historical questions."""
    historia = doc.get_section("Historia")
    etnobotanica = doc.get_section("Etnobotánica")

    if not historia or not etnobotanica:
        return "", []

    prompt = SYNTHESIS_PROMPTS[3]["prompt"].format(
        plant_name=doc.title,
        historia=historia[:500],
        etnobotanica=etnobotanica[:500],
    )
    return prompt, [doc.file_path]


def create_llm_prompt_for_safety(doc: ParsedDocument) -> tuple[str, list[str]]:
    """Create a prompt for generating safety questions."""
    toxicidad = doc.get_section("Toxicidad")
    etnobotanica = doc.get_section("Etnobotánica")

    if not toxicidad:
        return "", []

    prompt = SYNTHESIS_PROMPTS[4]["prompt"].format(
        plant_name=doc.title,
        toxicidad=toxicidad[:500],
        etnobotanica=(etnobotanica or "")[:300],
    )
    return prompt, [doc.file_path]


def call_anthropic_api(prompt: str, api_key: str) -> str | None:
    """Call the Anthropic API to generate content."""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except ImportError:
        print("anthropic package not installed. Run: pip install anthropic")
        return None
    except Exception as e:
        print(f"API error: {e}")
        return None


def call_openai_api(prompt: str, api_key: str) -> str | None:
    """Call the OpenAI API to generate content."""
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except ImportError:
        print("openai package not installed. Run: pip install openai")
        return None
    except Exception as e:
        print(f"API error: {e}")
        return None


def parse_llm_response(response: str) -> tuple[str, str, str] | None:
    """Parse the LLM response to extract question, reasoning, and answer."""
    try:
        lines = response.strip().split("\n")
        question = ""
        reasoning = ""
        answer = ""
        current_section = None

        for line in lines:
            if line.startswith("PREGUNTA:"):
                current_section = "question"
                question = line[9:].strip()
            elif line.startswith("RAZONAMIENTO:"):
                current_section = "reasoning"
                reasoning = line[13:].strip()
            elif line.startswith("RESPUESTA:"):
                current_section = "answer"
                answer = line[10:].strip()
            elif current_section == "question":
                question += " " + line.strip()
            elif current_section == "reasoning":
                reasoning += " " + line.strip()
            elif current_section == "answer":
                answer += " " + line.strip()

        if question and reasoning and answer:
            return question.strip(), reasoning.strip(), answer.strip()
        return None
    except Exception:
        return None


def generate_augmented_samples(
    documents: list[ParsedDocument],
    api_key: str,
    provider: str = "anthropic",
    max_samples: int = 100,
    seed: int = 42,
) -> list[AugmentedSample]:
    """Generate LLM-augmented training samples."""
    random.seed(seed)
    samples = []

    # Separate by document type
    plantas_docs = [d for d in documents if d.doc_type == "plantas"]
    diccionario_docs = [d for d in documents if d.doc_type == "diccionario"]

    # Shuffle for random selection
    random.shuffle(plantas_docs)
    random.shuffle(diccionario_docs)

    call_api = call_anthropic_api if provider == "anthropic" else call_openai_api

    # Generate different types of questions
    generators = [
        ("comparison", lambda: create_llm_prompt_for_comparison(plantas_docs)),
        ("mechanism", lambda: create_llm_prompt_for_mechanism(random.choice(plantas_docs))),
        ("cultural", lambda: create_llm_prompt_for_cultural(random.choice(diccionario_docs))),
        ("historical", lambda: create_llm_prompt_for_historical(random.choice(plantas_docs))),
        ("safety", lambda: create_llm_prompt_for_safety(random.choice(plantas_docs))),
    ]

    samples_per_type = max_samples // len(generators)

    for synthesis_type, generator_fn in generators:
        for _ in range(samples_per_type):
            prompt, source_files = generator_fn()
            if not prompt:
                continue

            response = call_api(prompt, api_key)
            if not response:
                continue

            parsed = parse_llm_response(response)
            if not parsed:
                continue

            question, reasoning, answer = parsed
            samples.append(
                AugmentedSample(
                    question=question,
                    reasoning=reasoning,
                    answer=answer,
                    synthesis_type=synthesis_type,
                    source_files=source_files,
                )
            )
            print(f"Generated {synthesis_type} question: {question[:50]}...")

    return samples


def main():
    """CLI entry point for LLM augmentation."""
    parser = argparse.ArgumentParser(
        description="Generate LLM-augmented training samples"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "data",
        help="Data directory containing markdown files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "augmented_samples.jsonl",
        help="Output file for augmented samples",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider to use",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of augmented samples to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling API",
    )
    args = parser.parse_args()

    # Get API key
    if args.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key and not args.dry_run:
            print("Error: ANTHROPIC_API_KEY environment variable not set")
            return
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key and not args.dry_run:
            print("Error: OPENAI_API_KEY environment variable not set")
            return

    print(f"Parsing documents from {args.data_dir}")
    documents = parse_all_documents(args.data_dir)
    print(f"Parsed {len(documents)} documents")

    if args.dry_run:
        # Just show example prompts
        plantas_docs = [d for d in documents if d.doc_type == "plantas"]
        diccionario_docs = [d for d in documents if d.doc_type == "diccionario"]
        random.seed(args.seed)
        random.shuffle(plantas_docs)
        random.shuffle(diccionario_docs)

        print("\n=== Example Prompts ===\n")

        prompt, _ = create_llm_prompt_for_comparison(plantas_docs)
        if prompt:
            print("--- Comparison ---")
            print(prompt[:500] + "...\n")

        if plantas_docs:
            prompt, _ = create_llm_prompt_for_mechanism(plantas_docs[0])
            if prompt:
                print("--- Mechanism ---")
                print(prompt[:500] + "...\n")

        if diccionario_docs:
            prompt, _ = create_llm_prompt_for_cultural(diccionario_docs[0])
            if prompt:
                print("--- Cultural ---")
                print(prompt[:500] + "...\n")

        return

    print(f"Generating augmented samples using {args.provider}...")
    samples = generate_augmented_samples(
        documents,
        api_key,
        provider=args.provider,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    print(f"Generated {len(samples)} augmented samples")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in samples:
            data = {
                "question": sample.question,
                "reasoning": sample.reasoning,
                "answer": sample.answer,
                "synthesis_type": sample.synthesis_type,
                "source_files": sample.source_files,
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Wrote samples to {args.output}")


if __name__ == "__main__":
    main()
