#!/usr/bin/env python3
"""
Audit script to detect gaps between scraped data and indexer expectations.

This script identifies:
1. Missing family fields in plant files
2. Navigation artifacts in content
3. Files with accented letter paths (for indexer regex check)
4. Invalid files (empty stems, whitespace-only titles)
5. Cross-reference link issues
6. Flora section status
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import yaml


DATA_DIR = Path(__file__).parent.parent / "data"

# Navigation artifact patterns to detect
NAV_PATTERNS = [
    r"BDMTM\s*>>\s*(?:APMTM|MTPIM|DEMTM|FMIM)",
    r"Inicio\s+Regresar\s+Imprimir",
    r"Biblioteca digital con fines de investigación y divulgación\.",
    r"Los conocimientos y la información original de esta publicación",
    r"está prohibida toda apropiación privada",
    r"2009 © D\.R\. Biblioteca Digital",
]

# Accented letters that should be matched by indexer
ACCENTED_LETTERS = set("áéíóúñčšüɨ")


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


def check_flora_section(data_dir: Path) -> dict:
    """Check the flora-medicinal section for completeness."""
    flora_dir = data_dir / "flora-medicinal"
    result = {
        "exists": flora_dir.exists(),
        "file_count": 0,
        "files": [],
        "status": "CRITICAL",
    }

    if flora_dir.exists():
        files = list(flora_dir.glob("*.md"))
        result["file_count"] = len(files)
        result["files"] = [f.name for f in files]

        if len(files) > 10:
            result["status"] = "OK"
        elif len(files) > 1:
            result["status"] = "WARNING"

    return result


def check_plant_families(data_dir: Path) -> dict:
    """Check for missing family fields in plant files."""
    plantas_dir = data_dir / "plantas"
    result = {
        "total_files": 0,
        "with_family": 0,
        "without_family": [],
        "percentage": 0.0,
    }

    if not plantas_dir.exists():
        return result

    for md_file in plantas_dir.rglob("*.md"):
        result["total_files"] += 1
        content = md_file.read_text(encoding="utf-8")
        metadata, _ = parse_frontmatter(content)

        if metadata.get("family"):
            result["with_family"] += 1
        else:
            result["without_family"].append(str(md_file.relative_to(data_dir)))

    if result["total_files"] > 0:
        result["percentage"] = (result["with_family"] / result["total_files"]) * 100

    return result


def check_accented_paths(data_dir: Path) -> dict:
    """Find files in folders with accented letters."""
    result = {
        "accented_folders": [],
        "file_count": 0,
        "affected_files": [],
    }

    diccionario_dir = data_dir / "diccionario"
    if not diccionario_dir.exists():
        return result

    for subdir in diccionario_dir.iterdir():
        if subdir.is_dir() and len(subdir.name) == 1:
            if subdir.name in ACCENTED_LETTERS or subdir.name not in "abcdefghijklmnopqrstuvwxyz":
                if subdir.name != "misc":
                    result["accented_folders"].append(subdir.name)
                    files = list(subdir.glob("*.md"))
                    result["file_count"] += len(files)
                    result["affected_files"].extend(
                        [str(f.relative_to(data_dir)) for f in files[:5]]
                    )

    return result


def check_navigation_artifacts(data_dir: Path) -> dict:
    """Detect navigation text artifacts in content."""
    result = {
        "total_checked": 0,
        "affected_files": [],
        "patterns_found": defaultdict(int),
    }

    # Check pueblos-indigenas and flora-medicinal primarily
    dirs_to_check = [
        data_dir / "pueblos-indigenas",
        data_dir / "flora-medicinal",
    ]

    for check_dir in dirs_to_check:
        if not check_dir.exists():
            continue

        for md_file in check_dir.glob("*.md"):
            result["total_checked"] += 1
            content = md_file.read_text(encoding="utf-8")
            _, body = parse_frontmatter(content)

            file_has_artifacts = False
            for pattern in NAV_PATTERNS:
                if re.search(pattern, body):
                    result["patterns_found"][pattern] += 1
                    file_has_artifacts = True

            if file_has_artifacts:
                result["affected_files"].append(str(md_file.relative_to(data_dir)))

    return result


def check_invalid_files(data_dir: Path) -> dict:
    """Find files with invalid names or empty/whitespace titles."""
    result = {
        "empty_stems": [],
        "whitespace_titles": [],
        "spaces_in_names": [],
    }

    for md_file in data_dir.rglob("*.md"):
        # Check for empty stem (.md only)
        if md_file.stem == "":
            result["empty_stems"].append(str(md_file.relative_to(data_dir)))
            continue

        # Check for spaces in filename
        if " " in md_file.name:
            result["spaces_in_names"].append(str(md_file.relative_to(data_dir)))

        # Check for whitespace-only titles
        content = md_file.read_text(encoding="utf-8")
        metadata, _ = parse_frontmatter(content)
        title = metadata.get("title", "")
        if isinstance(title, str) and (not title or title.strip() == ""):
            result["whitespace_titles"].append(str(md_file.relative_to(data_dir)))

    return result


def check_cross_references(data_dir: Path) -> dict:
    """Check for broken cross-reference links."""
    result = {
        "total_links": 0,
        "broken_links": [],
    }

    # Sample check - look for markdown links in a few files
    plantas_dir = data_dir / "plantas"
    if not plantas_dir.exists():
        return result

    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    for md_file in list(plantas_dir.rglob("*.md"))[:100]:  # Sample first 100
        content = md_file.read_text(encoding="utf-8")
        _, body = parse_frontmatter(content)

        for match in link_pattern.finditer(body):
            link_text, link_target = match.groups()
            result["total_links"] += 1

            # Check if it's a relative link that should resolve
            if link_target.startswith("../") or link_target.startswith("./"):
                target_path = (md_file.parent / link_target).resolve()
                if not target_path.exists():
                    # Check for spaces issue
                    if " " in link_target:
                        result["broken_links"].append({
                            "file": str(md_file.relative_to(data_dir)),
                            "link": link_target,
                            "reason": "spaces_in_path",
                        })

    return result


def print_report(results: dict) -> None:
    """Print a formatted audit report."""
    print("\n" + "=" * 60)
    print("HERBOLARIA DATA AUDIT REPORT")
    print("=" * 60)

    # Flora section
    print("\n## 1. Flora Section Status")
    flora = results["flora"]
    print(f"   Status: {flora['status']}")
    print(f"   Files found: {flora['file_count']}")
    if flora["files"]:
        print(f"   Files: {', '.join(flora['files'][:5])}")
    if flora["status"] != "OK":
        print("   ACTION: Flora scraper needs debugging - expected 10+ monographs")

    # Plant families
    print("\n## 2. Plant Family Fields")
    families = results["families"]
    print(f"   Total plant files: {families['total_files']}")
    print(f"   With family field: {families['with_family']} ({families['percentage']:.1f}%)")
    print(f"   Missing family: {len(families['without_family'])}")
    if families["percentage"] < 10:
        print("   ACTION: Family extraction in scraper needs improvement")

    # Accented paths
    print("\n## 3. Accented Letter Folders")
    accented = results["accented"]
    print(f"   Accented folders: {', '.join(accented['accented_folders'])}")
    print(f"   Files in accented folders: {accented['file_count']}")
    if accented["file_count"] > 0:
        print("   ACTION: Indexer regex needs to handle accented letters")
        print(f"   Sample files: {accented['affected_files'][:3]}")

    # Navigation artifacts
    print("\n## 4. Navigation Artifacts")
    nav = results["navigation"]
    print(f"   Files checked: {nav['total_checked']}")
    print(f"   Files with artifacts: {len(nav['affected_files'])}")
    if nav["affected_files"]:
        print(f"   Sample affected: {nav['affected_files'][:3]}")
        print("   ACTION: Cleanup needed for pueblos-indigenas files")

    # Invalid files
    print("\n## 5. Invalid Files")
    invalid = results["invalid"]
    print(f"   Empty stem files (.md): {len(invalid['empty_stems'])}")
    for f in invalid["empty_stems"]:
        print(f"      - {f}")
    print(f"   Whitespace-only titles: {len(invalid['whitespace_titles'])}")
    for f in invalid["whitespace_titles"][:5]:
        print(f"      - {f}")
    print(f"   Files with spaces in name: {len(invalid['spaces_in_names'])}")
    for f in invalid["spaces_in_names"][:3]:
        print(f"      - {f}")
    if invalid["empty_stems"] or invalid["whitespace_titles"]:
        print("   ACTION: Delete invalid files")

    # Cross-references
    print("\n## 6. Cross-References")
    xref = results["cross_refs"]
    print(f"   Links checked (sample): {xref['total_links']}")
    print(f"   Potentially broken: {len(xref['broken_links'])}")

    print("\n" + "=" * 60)
    print("END OF REPORT")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Audit Herbolaria data for gaps")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Data directory to audit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    print(f"Auditing data directory: {args.data_dir}")

    results = {
        "flora": check_flora_section(args.data_dir),
        "families": check_plant_families(args.data_dir),
        "accented": check_accented_paths(args.data_dir),
        "navigation": check_navigation_artifacts(args.data_dir),
        "invalid": check_invalid_files(args.data_dir),
        "cross_refs": check_cross_references(args.data_dir),
    }

    if args.json:
        import json
        # Convert defaultdict to regular dict for JSON
        results["navigation"]["patterns_found"] = dict(
            results["navigation"]["patterns_found"]
        )
        print(json.dumps(results, indent=2))
    else:
        print_report(results)


if __name__ == "__main__":
    main()
