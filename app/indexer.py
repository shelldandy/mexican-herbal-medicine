"""Document loading and index building for the Herbolaria RAG application."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_DIR = Path(__file__).parent.parent / "index"
MANIFEST_FILE = INDEX_DIR / "manifest.json"

SECTION_MAP = {
    "diccionario": "dictionary",
    "plantas": "plants",
    "pueblos-indigenas": "peoples",
    "flora-medicinal": "flora",
}


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from markdown content."""
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


def detect_section(file_path: Path) -> str:
    """Detect which section a file belongs to based on its path."""
    relative = file_path.relative_to(DATA_DIR)
    top_dir = relative.parts[0] if relative.parts else ""
    return SECTION_MAP.get(top_dir, "unknown")


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of file contents for change detection."""
    return hashlib.md5(file_path.read_bytes()).hexdigest()


def load_manifest() -> dict:
    """Load the manifest of indexed files."""
    if MANIFEST_FILE.exists():
        return json.loads(MANIFEST_FILE.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    """Save the manifest of indexed files."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Load all markdown documents from the data directory."""
    documents = []

    md_files = list(data_dir.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    for file_path in md_files:
        if file_path.name.startswith("."):
            continue

        content = file_path.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(content)

        section = detect_section(file_path)
        relative_path = str(file_path.relative_to(data_dir))

        letter_match = re.search(r"/([a-z])/", relative_path)
        letter = letter_match.group(1) if letter_match else ""

        doc_metadata = {
            "file_path": relative_path,
            "section": section,
            "title": metadata.get("title", file_path.stem),
            "source": metadata.get("source", ""),
            "letter": letter,
        }

        if "botanical_name" in metadata:
            doc_metadata["botanical_name"] = metadata["botanical_name"]
        if "family" in metadata:
            doc_metadata["family"] = metadata["family"]

        doc = Document(text=body, metadata=doc_metadata)
        documents.append(doc)

    return documents


def build_index(
    documents: list[Document],
    api_key: str | None = None,
    show_progress: bool = True,
) -> VectorStoreIndex:
    """Build the vector index from documents."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if api_key:
        embed_model = OpenAIEmbedding(api_key=api_key)
    else:
        embed_model = OpenAIEmbedding()

    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 128

    node_parser = MarkdownNodeParser()

    index = VectorStoreIndex.from_documents(
        documents,
        node_parser=node_parser,
        show_progress=show_progress,
    )

    index.storage_context.persist(persist_dir=str(INDEX_DIR))

    manifest = {}
    for doc in documents:
        file_path = doc.metadata.get("file_path", "")
        full_path = DATA_DIR / file_path
        if full_path.exists():
            manifest[file_path] = compute_file_hash(full_path)
    save_manifest(manifest)

    return index


def load_index(api_key: str | None = None) -> VectorStoreIndex | None:
    """Load existing index from disk."""
    docstore_path = INDEX_DIR / "docstore.json"
    if not docstore_path.exists():
        return None

    if api_key:
        embed_model = OpenAIEmbedding(api_key=api_key)
    else:
        embed_model = OpenAIEmbedding()

    Settings.embed_model = embed_model

    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    return load_index_from_storage(storage_context)


def get_changed_files(data_dir: Path = DATA_DIR) -> tuple[list[Path], list[str]]:
    """Detect new or changed files since last index build."""
    manifest = load_manifest()
    new_or_changed = []
    deleted = []

    current_files = set()
    for file_path in data_dir.rglob("*.md"):
        if file_path.name.startswith("."):
            continue

        relative = str(file_path.relative_to(data_dir))
        current_files.add(relative)

        current_hash = compute_file_hash(file_path)
        if relative not in manifest or manifest[relative] != current_hash:
            new_or_changed.append(file_path)

    for indexed_file in manifest:
        if indexed_file not in current_files:
            deleted.append(indexed_file)

    return new_or_changed, deleted


def main():
    """CLI entry point for building the index."""
    parser = argparse.ArgumentParser(
        description="Build or update the Herbolaria RAG index"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the full index from scratch",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check for new or changed files without building",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Data directory containing markdown files",
    )
    args = parser.parse_args()

    if args.check:
        new_or_changed, deleted = get_changed_files(args.data_dir)
        print(f"New or changed files: {len(new_or_changed)}")
        print(f"Deleted files: {len(deleted)}")
        for f in new_or_changed[:10]:
            print(f"  + {f}")
        if len(new_or_changed) > 10:
            print(f"  ... and {len(new_or_changed) - 10} more")
        return

    if args.build:
        print("Loading documents...")
        documents = load_documents(args.data_dir)
        print(f"Loaded {len(documents)} documents")

        print("Building index (this requires OPENAI_API_KEY env var)...")
        build_index(documents, show_progress=True)
        print(f"Index built successfully at {INDEX_DIR}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
