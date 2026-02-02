#!/usr/bin/env python3
"""
Cleanup script to fix data issues identified by audit.

This script:
1. Removes invalid files (empty stems, whitespace-only titles)
2. Normalizes filenames with spaces to use hyphens
3. Removes navigation artifacts from content
4. Updates cross-references after filename changes
"""

import argparse
import re
import shutil
from pathlib import Path

import yaml


DATA_DIR = Path(__file__).parent.parent / "data"

# Navigation artifact patterns to remove
NAV_PATTERNS = [
    # Full navigation breadcrumb lines
    r"^BDMTM\s*>>\s*(?:APMTM|MTPIM|DEMTM|FMIM).*?(?:Inicio|Imprimir).*$",
    r"^BDMTM\s*$",
    r"^>>\s*$",
    r"^(?:APMTM|MTPIM|DEMTM|FMIM)\s*$",
    r"^Inicio\s*$",
    r"^Regresar\s*$",
    r"^Imprimir\s*$",
    # Footer disclaimers (often at end of content)
    r"Biblioteca digital con fines de investigación y divulgación\. No tiene la intención de ofrecer prescripciones médicas\. El uso que se dé a la información contenida en este sitio es responsabilidad estricta del lector\.",
    r"Los conocimientos y la información original de esta publicación son de origen y creación colectiva, sus poseedores y recreadores son los pueblos indígenas de México, por lo que deben seguir siendo colectivos y, en consecuencia, está prohibida toda apropiación privada\.",
    r"2009 © D\.R\. Biblioteca Digital de la Medicina Tradicional Mexicana\. Hecho en México\.",
]

# Inline navigation patterns (within paragraphs)
INLINE_NAV_PATTERNS = [
    r"BDMTM\s*>>\s*(?:APMTM|MTPIM|DEMTM|FMIM)\s*>>\s*[^>]+>>\s*(?:Población|[^>]+)\s*Inicio\s+Regresar\s+Imprimir\s*",
    r"\[\s*La población\s*\]\s*\[\s*Los recursos humanos\s*\]\s*\[\s*Las demandas de atención\s*\]\s*\[\s*Descripción de demandas\s*\]",
]


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

    body = parts[2]
    return metadata, body


def generate_frontmatter(metadata: dict) -> str:
    """Generate YAML frontmatter string."""
    return "---\n" + yaml.dump(metadata, allow_unicode=True, default_flow_style=False) + "---\n"


def normalize_filename(name: str) -> str:
    """Normalize filename by replacing spaces and commas with hyphens."""
    normalized = name.replace(" ", "-").replace(",", "")
    # Clean up multiple consecutive hyphens
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized


def remove_navigation_artifacts(content: str) -> str:
    """Remove navigation text artifacts from content."""
    metadata, body = parse_frontmatter(content)

    # Remove inline navigation patterns first (these are within paragraphs)
    for pattern in INLINE_NAV_PATTERNS:
        body = re.sub(pattern, "", body, flags=re.MULTILINE | re.DOTALL)

    # Remove footer disclaimers that appear embedded in paragraphs
    footer_patterns = [
        r"\s*Biblioteca digital con fines de investigación y divulgación\.[^.]*\.",
        r"\s*Los conocimientos y la información original de esta publicación[^.]*\.",
        r"\s*2009 © D\.R\. Biblioteca Digital de la Medicina Tradicional Mexicana\. Hecho en México\.",
    ]
    for pattern in footer_patterns:
        body = re.sub(pattern, "", body, flags=re.MULTILINE | re.DOTALL)

    # Split into lines and filter
    lines = body.split("\n")
    cleaned_lines = []

    for line in lines:
        line_stripped = line.strip()

        # Skip empty lines that would create multiple consecutive blank lines
        if not line_stripped:
            if cleaned_lines and cleaned_lines[-1].strip() == "":
                continue

        # Check against navigation patterns
        skip_line = False
        for pattern in NAV_PATTERNS:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                skip_line = True
                break

        if not skip_line:
            cleaned_lines.append(line)

    # Clean up trailing whitespace and excessive newlines
    body = "\n".join(cleaned_lines)

    # For pueblos-indigenas files: remove duplicate sections
    # These files often have the same content repeated 3-4 times
    sections = body.split("\n## ")
    if len(sections) > 1:
        unique_sections = [sections[0]]
        seen_content = set()
        for section in sections[1:]:
            # Use first 300 chars as signature to detect duplicates
            # Normalize whitespace for comparison
            normalized = " ".join(section[:300].split())
            if normalized not in seen_content:
                seen_content.add(normalized)
                unique_sections.append(section)
        body = "\n## ".join(unique_sections)

    # For content before first section heading: detect and remove duplicates
    # This handles the case where the intro content is repeated
    first_section_idx = body.find("\n## ")
    if first_section_idx > 0:
        intro = body[:first_section_idx]
        rest = body[first_section_idx:]

        # Split intro by double newlines into paragraphs
        paragraphs = [p.strip() for p in intro.split("\n\n") if p.strip()]
        unique_paragraphs = []
        seen_para = set()

        for para in paragraphs:
            # Normalize for comparison
            normalized = " ".join(para[:200].split())
            if normalized not in seen_para:
                seen_para.add(normalized)
                unique_paragraphs.append(para)

        body = "\n\n".join(unique_paragraphs) + rest

    # Clean up multiple consecutive blank lines
    while "\n\n\n" in body:
        body = body.replace("\n\n\n", "\n\n")

    if metadata:
        return generate_frontmatter(metadata) + "\n" + body.strip() + "\n"
    return body.strip() + "\n"


def cleanup_invalid_files(data_dir: Path, dry_run: bool = True) -> dict:
    """Remove invalid files (empty stems, whitespace-only titles)."""
    result = {"deleted": [], "errors": []}

    for md_file in data_dir.rglob("*.md"):
        should_delete = False
        reason = ""

        # Check for empty stem
        if not md_file.stem:
            should_delete = True
            reason = "empty_stem"

        # Check for whitespace-only titles
        if not should_delete:
            try:
                content = md_file.read_text(encoding="utf-8")
                metadata, _ = parse_frontmatter(content)
                title = metadata.get("title", "")
                if isinstance(title, str) and not title.strip():
                    should_delete = True
                    reason = "whitespace_title"
            except Exception as e:
                result["errors"].append({"file": str(md_file), "error": str(e)})
                continue

        if should_delete:
            relative = str(md_file.relative_to(data_dir))
            if dry_run:
                print(f"Would delete: {relative} ({reason})")
            else:
                try:
                    md_file.unlink()
                    print(f"Deleted: {relative} ({reason})")
                    result["deleted"].append(relative)
                except Exception as e:
                    result["errors"].append({"file": relative, "error": str(e)})

    return result


def normalize_filenames(data_dir: Path, dry_run: bool = True) -> dict:
    """Normalize filenames with spaces to use hyphens."""
    result = {"renamed": [], "errors": []}

    # Focus on pueblos-indigenas and images/pueblos directories
    dirs_to_check = [
        data_dir / "pueblos-indigenas",
        data_dir / "images" / "pueblos",
    ]

    for check_dir in dirs_to_check:
        if not check_dir.exists():
            continue

        for file_path in list(check_dir.iterdir()):
            if " " in file_path.name or "," in file_path.name:
                new_name = normalize_filename(file_path.name)
                new_path = file_path.parent / new_name

                old_relative = str(file_path.relative_to(data_dir))
                new_relative = str(new_path.relative_to(data_dir))

                if dry_run:
                    print(f"Would rename: {old_relative} -> {new_relative}")
                else:
                    try:
                        file_path.rename(new_path)
                        print(f"Renamed: {old_relative} -> {new_relative}")
                        result["renamed"].append({
                            "old": old_relative,
                            "new": new_relative,
                        })
                    except Exception as e:
                        result["errors"].append({
                            "file": old_relative,
                            "error": str(e),
                        })

    return result


def cleanup_navigation_artifacts(data_dir: Path, dry_run: bool = True) -> dict:
    """Remove navigation artifacts from content files."""
    result = {"cleaned": [], "errors": []}

    # Focus on pueblos-indigenas directory
    pueblos_dir = data_dir / "pueblos-indigenas"
    if not pueblos_dir.exists():
        return result

    for md_file in pueblos_dir.glob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            cleaned = remove_navigation_artifacts(content)

            if cleaned != content:
                relative = str(md_file.relative_to(data_dir))
                if dry_run:
                    # Show size reduction
                    reduction = len(content) - len(cleaned)
                    print(f"Would clean: {relative} (reduce by {reduction} chars)")
                else:
                    md_file.write_text(cleaned, encoding="utf-8")
                    print(f"Cleaned: {relative}")
                    result["cleaned"].append(relative)

        except Exception as e:
            result["errors"].append({
                "file": str(md_file.relative_to(data_dir)),
                "error": str(e),
            })

    return result


def update_cross_references(data_dir: Path, renames: list[dict], dry_run: bool = True) -> dict:
    """Update cross-references in markdown files after renames."""
    result = {"updated": [], "errors": []}

    if not renames:
        return result

    # Build a mapping of old paths to new paths
    path_map = {}
    for rename in renames:
        old_name = Path(rename["old"]).name
        new_name = Path(rename["new"]).name
        path_map[old_name] = new_name

    # Check all markdown files for references
    for md_file in data_dir.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            updated_content = content

            for old_name, new_name in path_map.items():
                # Update markdown links
                updated_content = updated_content.replace(f"]({old_name})", f"]({new_name})")
                updated_content = updated_content.replace(f"](../{old_name})", f"](../{new_name})")

                # Update image references
                old_stem = Path(old_name).stem
                new_stem = Path(new_name).stem
                updated_content = re.sub(
                    rf"!\[([^\]]*)\]\(([^)]*){re.escape(old_stem)}([^)]*)\)",
                    rf"![\1](\2{new_stem}\3)",
                    updated_content,
                )

            if updated_content != content:
                relative = str(md_file.relative_to(data_dir))
                if dry_run:
                    print(f"Would update refs in: {relative}")
                else:
                    md_file.write_text(updated_content, encoding="utf-8")
                    print(f"Updated refs in: {relative}")
                    result["updated"].append(relative)

        except Exception as e:
            result["errors"].append({
                "file": str(md_file.relative_to(data_dir)),
                "error": str(e),
            })

    return result


def main():
    parser = argparse.ArgumentParser(description="Clean up Herbolaria data issues")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Data directory to clean",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--delete-invalid",
        action="store_true",
        help="Delete invalid files (empty stems, whitespace titles)",
    )
    parser.add_argument(
        "--normalize-names",
        action="store_true",
        help="Normalize filenames (replace spaces with hyphens)",
    )
    parser.add_argument(
        "--clean-navigation",
        action="store_true",
        help="Remove navigation artifacts from content",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all cleanup operations",
    )
    args = parser.parse_args()

    if not any([args.delete_invalid, args.normalize_names, args.clean_navigation, args.all]):
        parser.print_help()
        print("\nSpecify at least one cleanup operation or use --all")
        return

    print(f"Data directory: {args.data_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    renames = []

    if args.delete_invalid or args.all:
        print("=" * 40)
        print("Cleaning invalid files...")
        print("=" * 40)
        cleanup_invalid_files(args.data_dir, dry_run=args.dry_run)
        print()

    if args.normalize_names or args.all:
        print("=" * 40)
        print("Normalizing filenames...")
        print("=" * 40)
        result = normalize_filenames(args.data_dir, dry_run=args.dry_run)
        renames = result.get("renamed", [])
        print()

    if args.clean_navigation or args.all:
        print("=" * 40)
        print("Cleaning navigation artifacts...")
        print("=" * 40)
        cleanup_navigation_artifacts(args.data_dir, dry_run=args.dry_run)
        print()

    # Update cross-references after renames
    if renames and not args.dry_run:
        print("=" * 40)
        print("Updating cross-references...")
        print("=" * 40)
        update_cross_references(args.data_dir, renames, dry_run=False)
        print()

    if args.dry_run:
        print("Dry run complete. Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
