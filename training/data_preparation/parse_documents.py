"""Parse markdown documents and extract structured sections.

This module reuses parsing patterns from app/indexer.py for consistent
frontmatter and section detection across the herbolaria codebase.
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Section:
    """A section within a document."""

    name: str
    content: str
    level: int = 2


@dataclass
class ParsedDocument:
    """A parsed document with metadata and sections."""

    file_path: str
    doc_type: str  # plantas, diccionario, pueblos-indigenas
    title: str
    botanical_name: str | None = None
    family: str | None = None
    source: str | None = None
    sections: list[Section] = field(default_factory=list)
    raw_body: str = ""

    def get_section(self, name: str) -> str | None:
        """Get content of a section by name (case-insensitive partial match)."""
        name_lower = name.lower()
        for section in self.sections:
            if name_lower in section.name.lower():
                return section.content
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "doc_type": self.doc_type,
            "title": self.title,
            "botanical_name": self.botanical_name,
            "family": self.family,
            "source": self.source,
            "sections": [
                {"name": s.name, "content": s.content, "level": s.level}
                for s in self.sections
            ],
        }


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from markdown content.

    Reused from app/indexer.py:32-47.
    """
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    try:
        metadata = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        metadata = {}

    body = parts[2].strip()
    return metadata, body


def detect_doc_type(file_path: Path, data_dir: Path) -> str:
    """Detect document type based on path.

    Based on app/indexer.py:50-54 section detection.
    """
    try:
        relative = file_path.relative_to(data_dir)
        top_dir = relative.parts[0] if relative.parts else ""
        type_map = {
            "diccionario": "diccionario",
            "plantas": "plantas",
            "pueblos-indigenas": "pueblos-indigenas",
            "flora-medicinal": "flora-medicinal",
        }
        return type_map.get(top_dir, "unknown")
    except ValueError:
        return "unknown"


def extract_sections(body: str) -> list[Section]:
    """Extract markdown sections from document body.

    Identifies H2 (##) and H3 (###) sections and extracts their content.
    """
    sections = []

    # Pattern to match markdown headers
    header_pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)

    matches = list(header_pattern.finditer(body))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        name = match.group(2).strip()

        # Get content from after this header to before the next header
        start = match.end()
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(body)

        content = body[start:end].strip()

        # Skip empty sections
        if content:
            sections.append(Section(name=name, content=content, level=level))

    return sections


def parse_document(file_path: Path, data_dir: Path) -> ParsedDocument:
    """Parse a single markdown document.

    Args:
        file_path: Path to the markdown file.
        data_dir: Base data directory for relative path calculation.

    Returns:
        ParsedDocument with extracted metadata and sections.
    """
    content = file_path.read_text(encoding="utf-8")
    metadata, body = parse_frontmatter(content)

    doc_type = detect_doc_type(file_path, data_dir)

    # Extract title from metadata or filename
    title = metadata.get("title", file_path.stem.replace("-", " ").title())

    # Extract botanical name if present (for plantas)
    botanical_name = metadata.get("botanical_name")

    # Extract sections
    sections = extract_sections(body)

    # Calculate relative path
    try:
        rel_path = str(file_path.relative_to(data_dir))
    except ValueError:
        rel_path = str(file_path)

    return ParsedDocument(
        file_path=rel_path,
        doc_type=doc_type,
        title=title,
        botanical_name=botanical_name,
        family=metadata.get("family"),
        source=metadata.get("source"),
        sections=sections,
        raw_body=body,
    )


def parse_all_documents(data_dir: Path) -> list[ParsedDocument]:
    """Parse all markdown documents in the data directory.

    Args:
        data_dir: Path to the data directory.

    Returns:
        List of parsed documents.
    """
    documents = []

    md_files = list(data_dir.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    for file_path in md_files:
        # Skip hidden files
        if file_path.name.startswith("."):
            continue

        # Skip progress tracking files
        if ".progress" in str(file_path):
            continue

        try:
            doc = parse_document(file_path, data_dir)
            documents.append(doc)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            continue

    return documents


def main():
    """CLI entry point for parsing documents."""
    parser = argparse.ArgumentParser(
        description="Parse herbolaria markdown documents for training data extraction"
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
        default=Path(__file__).parent.parent / "parsed_documents.jsonl",
        help="Output file for parsed documents (JSONL format)",
    )
    parser.add_argument(
        "--doc-type",
        choices=["plantas", "diccionario", "pueblos-indigenas", "flora-medicinal"],
        help="Filter by document type",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics about parsed documents",
    )
    args = parser.parse_args()

    print(f"Parsing documents from {args.data_dir}")
    documents = parse_all_documents(args.data_dir)

    # Filter by type if specified
    if args.doc_type:
        documents = [d for d in documents if d.doc_type == args.doc_type]

    print(f"Parsed {len(documents)} documents")

    if args.stats:
        # Print statistics
        by_type = {}
        section_counts = {}
        for doc in documents:
            by_type[doc.doc_type] = by_type.get(doc.doc_type, 0) + 1
            for section in doc.sections:
                section_counts[section.name] = section_counts.get(section.name, 0) + 1

        print("\nDocuments by type:")
        for doc_type, count in sorted(by_type.items()):
            print(f"  {doc_type}: {count}")

        print("\nMost common sections:")
        for name, count in sorted(
            section_counts.items(), key=lambda x: -x[1]
        )[:20]:
            print(f"  {name}: {count}")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

    print(f"Wrote {len(documents)} documents to {args.output}")


if __name__ == "__main__":
    main()
