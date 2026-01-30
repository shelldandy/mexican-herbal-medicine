"""Generate Q&A pairs from parsed documents using templates.

This module creates instruction-response pairs for finetuning, with
chain-of-thought reasoning in the responses.
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .parse_documents import ParsedDocument, parse_all_documents

# Question templates organized by document type and section
QUESTION_TEMPLATES = {
    "plantas": {
        "general": [
            "¿Cuáles son los usos medicinales de {title}?",
            "¿Qué propiedades tiene {title} en la medicina tradicional mexicana?",
            "Describe las aplicaciones terapéuticas de {title}.",
            "¿Cómo se usa {title} para tratar enfermedades?",
            "¿Qué beneficios medicinales tiene la planta {title}?",
        ],
        "sinonimia_botanica": [
            "¿Cuáles son los sinónimos botánicos de {botanical_name}?",
            "¿Con qué otros nombres científicos se conoce a {title}?",
        ],
        "sinonimia_popular": [
            "¿Cuáles son los nombres populares de {title}?",
            "¿Cómo se le llama a {title} en diferentes regiones de México?",
            "¿Qué nombres indígenas tiene {title}?",
        ],
        "botanica_ecologia": [
            "Describe las características botánicas de {title}.",
            "¿Cómo es la planta {title} y dónde crece?",
            "¿En qué tipo de clima y vegetación se encuentra {title}?",
        ],
        "etnobotanica": [
            "¿Cuál es el uso etnobotánico de {title}?",
            "¿Cómo utilizan {title} los pueblos indígenas de México?",
            "Describe el uso tradicional de {title} en la medicina mexicana.",
            "¿Qué padecimientos se tratan con {title} en la medicina tradicional?",
        ],
        "historia": [
            "¿Cuál es la historia del uso medicinal de {title}?",
            "¿Cómo se usaba {title} en la época prehispánica y colonial?",
            "¿Qué dicen las fuentes históricas sobre {title}?",
        ],
        "quimica": [
            "¿Qué compuestos químicos contiene {title}?",
            "¿Cuáles son los componentes activos de {botanical_name}?",
            "Describe la composición química de {title}.",
        ],
        "farmacologia": [
            "¿Qué estudios farmacológicos se han realizado sobre {title}?",
            "¿Qué efectos farmacológicos tiene {botanical_name}?",
            "¿Cómo actúa {title} según los estudios científicos?",
        ],
        "principios_activos": [
            "¿Cuáles son los principios activos de {title}?",
            "¿Qué sustancias dan las propiedades medicinales a {title}?",
        ],
        "toxicidad": [
            "¿Cuál es la toxicidad de {title}?",
            "¿Qué precauciones hay que tomar al usar {title}?",
            "¿Es seguro usar {title}? ¿Tiene efectos adversos?",
            "¿Existen contraindicaciones para el uso de {title}?",
        ],
    },
    "diccionario": {
        "general": [
            "¿Qué es {title} en la medicina tradicional mexicana?",
            "Define {title} según la medicina tradicional.",
            "Explica el concepto de {title} en el contexto de la medicina tradicional mexicana.",
        ],
        "descripcion": [
            "¿Cómo se describe {title} en la medicina tradicional?",
            "¿Qué significa {title} y cuáles son sus sinónimos?",
        ],
        "tratamiento": [
            "¿Cómo se trata {title} en la medicina tradicional?",
            "¿Qué remedios tradicionales existen para {title}?",
        ],
    },
    "pueblos-indigenas": {
        "general": [
            "¿Cómo es la medicina tradicional del pueblo {title}?",
            "Describe las prácticas médicas tradicionales de los {title}.",
            "¿Qué caracteriza la medicina tradicional de los {title}?",
        ],
        "poblacion": [
            "¿Dónde se ubica el pueblo {title} en México?",
            "¿Cuál es la distribución geográfica de los {title}?",
        ],
        "recursos_humanos": [
            "¿Qué tipos de curanderos o terapeutas tienen los {title}?",
            "¿Cómo se organizan los médicos tradicionales entre los {title}?",
        ],
        "demandas_atencion": [
            "¿Cuáles son las principales enfermedades que atiende la medicina tradicional {title}?",
            "¿Qué padecimientos tratan los curanderos {title}?",
        ],
    },
}

# Reasoning templates for chain-of-thought
REASONING_TEMPLATES = {
    "plantas_general": """Voy a analizar la información disponible sobre {title} ({botanical_name}) para responder esta pregunta sobre sus usos medicinales.

Primero, revisaré la información etnobotánica que describe los usos tradicionales documentados.
{etnobotanica_reasoning}

Luego, consideraré los estudios farmacológicos que validan algunos de estos usos.
{farmacologia_reasoning}

Finalmente, debo mencionar las precauciones importantes sobre toxicidad.
{toxicidad_reasoning}""",
    "plantas_quimica": """Para describir la composición química de {title}, analizaré los estudios fitoquímicos documentados.

Los compuestos identificados incluyen diferentes clases químicas que explican sus propiedades medicinales.
{quimica_content}""",
    "diccionario_general": """Para explicar {title} en el contexto de la medicina tradicional mexicana, debo considerar:

1. La definición y sinónimos en diferentes lenguas indígenas
2. El concepto dentro del sistema médico tradicional
3. Las creencias y prácticas asociadas

{descripcion_content}""",
}


@dataclass
class QAPair:
    """A question-answer pair for training."""

    question: str
    answer: str
    reasoning: str
    source_file: str
    doc_type: str
    section_type: str

    def to_chat_format(self, system_prompt: str) -> dict:
        """Convert to chat format with chain-of-thought."""
        # Combine reasoning and answer
        full_response = f"<pensamiento>\n{self.reasoning}\n</pensamiento>\n\n{self.answer}"

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.question},
                {"role": "assistant", "content": full_response},
            ],
            "metadata": {
                "source_file": self.source_file,
                "doc_type": self.doc_type,
                "section_type": self.section_type,
            },
        }


def generate_reasoning_for_plantas(doc: ParsedDocument, section_type: str) -> str:
    """Generate chain-of-thought reasoning for plant questions."""
    title = doc.title
    botanical_name = doc.botanical_name or title

    if section_type == "general":
        etnobotanica = doc.get_section("Etnobotánica") or ""
        farmacologia = doc.get_section("Farmacología") or ""
        toxicidad = doc.get_section("Toxicidad") or ""

        etnobotanica_reasoning = (
            f"La información etnobotánica indica varios usos tradicionales documentados en diferentes regiones de México."
            if etnobotanica
            else "No hay información etnobotánica detallada disponible."
        )

        farmacologia_reasoning = (
            f"Existen estudios farmacológicos que han investigado los efectos de esta planta."
            if farmacologia
            else "Los estudios farmacológicos son limitados."
        )

        toxicidad_reasoning = (
            f"Es importante considerar la información sobre toxicidad y contraindicaciones."
            if toxicidad
            else "No se reporta toxicidad significativa, pero se recomienda precaución."
        )

        return REASONING_TEMPLATES["plantas_general"].format(
            title=title,
            botanical_name=botanical_name,
            etnobotanica_reasoning=etnobotanica_reasoning,
            farmacologia_reasoning=farmacologia_reasoning,
            toxicidad_reasoning=toxicidad_reasoning,
        )

    elif section_type == "quimica":
        quimica = doc.get_section("Química") or "No hay información química detallada."
        return REASONING_TEMPLATES["plantas_quimica"].format(
            title=title,
            quimica_content=f"Según la literatura: {quimica[:500]}..."
            if len(quimica) > 500
            else quimica,
        )

    else:
        # Generic reasoning
        return f"Analizando la información disponible sobre {title} para responder sobre {section_type}."


def generate_reasoning_for_diccionario(doc: ParsedDocument) -> str:
    """Generate chain-of-thought reasoning for dictionary entries."""
    descripcion = doc.get_section("Descripción") or doc.raw_body[:1000]
    return REASONING_TEMPLATES["diccionario_general"].format(
        title=doc.title,
        descripcion_content=descripcion[:800] if len(descripcion) > 800 else descripcion,
    )


def generate_answer_from_section(doc: ParsedDocument, section_name: str) -> str:
    """Generate an answer from a specific section."""
    content = doc.get_section(section_name)
    if not content:
        return ""

    # Clean up the content
    answer = content.strip()

    # Add source citation
    if doc.source:
        answer += f"\n\n_Fuente: {doc.source}_"

    return answer


def generate_qa_pairs_for_plantas(doc: ParsedDocument) -> list[QAPair]:
    """Generate Q&A pairs for a plant document."""
    pairs = []
    title = doc.title
    botanical_name = doc.botanical_name or title

    templates = QUESTION_TEMPLATES["plantas"]

    # General questions using multiple sections
    etnobotanica = doc.get_section("Etnobotánica")
    if etnobotanica:
        for template in templates["general"][:2]:  # Limit to avoid duplicates
            question = template.format(title=title, botanical_name=botanical_name)
            reasoning = generate_reasoning_for_plantas(doc, "general")

            # Build comprehensive answer
            answer_parts = [f"**{title}** ({botanical_name})\n"]
            answer_parts.append(etnobotanica)

            farmacologia = doc.get_section("Farmacología")
            if farmacologia:
                answer_parts.append(f"\n\n**Estudios farmacológicos:**\n{farmacologia[:500]}")

            toxicidad = doc.get_section("Toxicidad")
            if toxicidad:
                answer_parts.append(f"\n\n**Precauciones:**\n{toxicidad[:300]}")

            answer = "\n".join(answer_parts)

            pairs.append(
                QAPair(
                    question=question,
                    answer=answer,
                    reasoning=reasoning,
                    source_file=doc.file_path,
                    doc_type="plantas",
                    section_type="general",
                )
            )

    # Section-specific questions
    section_mapping = {
        "sinonimia_botanica": "Sinonimia botánica",
        "sinonimia_popular": "Sinonimia popular",
        "botanica_ecologia": "Botánica y ecología",
        "etnobotanica": "Etnobotánica",
        "historia": "Historia",
        "quimica": "Química",
        "farmacologia": "Farmacología",
        "principios_activos": "Principios activos",
        "toxicidad": "Toxicidad",
    }

    for template_key, section_name in section_mapping.items():
        content = doc.get_section(section_name)
        if not content or len(content) < 50:  # Skip very short sections
            continue

        if template_key not in templates:
            continue

        # Pick one random template for this section
        template = random.choice(templates[template_key])
        question = template.format(title=title, botanical_name=botanical_name)
        reasoning = generate_reasoning_for_plantas(doc, template_key)
        answer = f"**{section_name} de {title}:**\n\n{content}"

        pairs.append(
            QAPair(
                question=question,
                answer=answer,
                reasoning=reasoning,
                source_file=doc.file_path,
                doc_type="plantas",
                section_type=template_key,
            )
        )

    return pairs


def generate_qa_pairs_for_diccionario(doc: ParsedDocument) -> list[QAPair]:
    """Generate Q&A pairs for a dictionary entry."""
    pairs = []
    title = doc.title

    templates = QUESTION_TEMPLATES["diccionario"]

    # Use raw body for dictionary entries as they may not have standard sections
    content = doc.get_section("Descripción") or doc.raw_body

    if not content or len(content) < 50:
        return pairs

    for template in templates["general"][:2]:
        question = template.format(title=title)
        reasoning = generate_reasoning_for_diccionario(doc)

        # Build answer
        answer = f"**{title}**\n\n{content}"

        pairs.append(
            QAPair(
                question=question,
                answer=answer,
                reasoning=reasoning,
                source_file=doc.file_path,
                doc_type="diccionario",
                section_type="general",
            )
        )

    return pairs


def generate_qa_pairs_for_pueblos(doc: ParsedDocument) -> list[QAPair]:
    """Generate Q&A pairs for indigenous peoples documents."""
    pairs = []
    title = doc.title

    templates = QUESTION_TEMPLATES["pueblos-indigenas"]

    # These documents are large, extract key sections
    for template_key, section_patterns in [
        ("poblacion", ["población", "La población"]),
        ("recursos_humanos", ["recursos humanos", "Los recursos"]),
        ("demandas_atencion", ["demandas de atención", "Las demandas"]),
    ]:
        content = None
        for pattern in section_patterns:
            content = doc.get_section(pattern)
            if content:
                break

        if not content or len(content) < 100:
            continue

        if template_key not in templates:
            continue

        template = random.choice(templates[template_key])
        question = template.format(title=title)
        reasoning = f"Analizando la información sobre {template_key.replace('_', ' ')} del pueblo {title}."
        answer = f"**{title} - {template_key.replace('_', ' ').title()}:**\n\n{content[:2000]}"

        pairs.append(
            QAPair(
                question=question,
                answer=answer,
                reasoning=reasoning,
                source_file=doc.file_path,
                doc_type="pueblos-indigenas",
                section_type=template_key,
            )
        )

    # General question
    if doc.raw_body and len(doc.raw_body) > 200:
        template = random.choice(templates["general"])
        question = template.format(title=title)
        reasoning = f"Voy a describir las características principales de la medicina tradicional del pueblo {title}."
        answer = f"**Medicina tradicional de los {title}:**\n\n{doc.raw_body[:2000]}"

        pairs.append(
            QAPair(
                question=question,
                answer=answer,
                reasoning=reasoning,
                source_file=doc.file_path,
                doc_type="pueblos-indigenas",
                section_type="general",
            )
        )

    return pairs


def generate_qa_pairs(documents: list[ParsedDocument]) -> list[QAPair]:
    """Generate Q&A pairs from all documents."""
    all_pairs = []

    for doc in documents:
        if doc.doc_type == "plantas":
            pairs = generate_qa_pairs_for_plantas(doc)
        elif doc.doc_type == "diccionario":
            pairs = generate_qa_pairs_for_diccionario(doc)
        elif doc.doc_type == "pueblos-indigenas":
            pairs = generate_qa_pairs_for_pueblos(doc)
        else:
            continue

        all_pairs.extend(pairs)

    return all_pairs


# Default system prompt for the assistant
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
    """CLI entry point for generating Q&A pairs."""
    parser = argparse.ArgumentParser(
        description="Generate Q&A pairs from herbolaria documents"
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
        default=Path(__file__).parent.parent / "training_samples.jsonl",
        help="Output file for Q&A pairs (JSONL format)",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "chat"],
        default="chat",
        help="Output format: jsonl (raw) or chat (ready for training)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Parsing documents from {args.data_dir}")
    documents = parse_all_documents(args.data_dir)
    print(f"Parsed {len(documents)} documents")

    print("Generating Q&A pairs...")
    pairs = generate_qa_pairs(documents)
    print(f"Generated {len(pairs)} Q&A pairs")

    # Shuffle and limit
    random.shuffle(pairs)
    if args.max_samples:
        pairs = pairs[: args.max_samples]

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            if args.format == "chat":
                data = pair.to_chat_format(SYSTEM_PROMPT)
            else:
                data = {
                    "question": pair.question,
                    "answer": pair.answer,
                    "reasoning": pair.reasoning,
                    "source_file": pair.source_file,
                    "doc_type": pair.doc_type,
                    "section_type": pair.section_type,
                }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Wrote {len(pairs)} samples to {args.output}")

    # Print statistics
    by_type = {}
    by_section = {}
    for pair in pairs:
        by_type[pair.doc_type] = by_type.get(pair.doc_type, 0) + 1
        by_section[pair.section_type] = by_section.get(pair.section_type, 0) + 1

    print("\nSamples by document type:")
    for doc_type, count in sorted(by_type.items()):
        print(f"  {doc_type}: {count}")

    print("\nSamples by section type:")
    for section_type, count in sorted(by_section.items(), key=lambda x: -x[1])[:10]:
        print(f"  {section_type}: {count}")


if __name__ == "__main__":
    main()
