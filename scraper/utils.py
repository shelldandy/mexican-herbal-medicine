"""Utility functions for scraping and markdown generation."""

import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs

from slugify import slugify
import yaml


BASE_URL = "http://www.medicinatradicionalmexicana.unam.mx"


def make_slug(text: str) -> str:
    """Create a URL-safe slug from text."""
    return slugify(text, lowercase=True)


def get_letter_from_term(term: str) -> str:
    """Extract the first letter (lowercased) from a term for folder organization."""
    if not term:
        return "misc"
    first_char = term[0].lower()
    if first_char.isalpha():
        return first_char
    return "misc"


def parse_internal_url(url: str) -> dict | None:
    """
    Parse an internal URL to determine its type and extract parameters.

    Returns dict with keys: section, slug, letter, subsection (if applicable)
    """
    if not url:
        return None

    # Make absolute if relative
    if url.startswith("/"):
        url = urljoin(BASE_URL, url)

    parsed = urlparse(url)

    # Must be from the same domain
    if parsed.netloc and parsed.netloc != "www.medicinatradicionalmexicana.unam.mx":
        return None

    path = parsed.path
    query = parse_qs(parsed.query)

    # Dictionary entry: /demtm/termino.php?l=1&t={slug}
    if "/demtm/termino.php" in path:
        slug = query.get("t", [""])[0]
        return {
            "section": "diccionario",
            "slug": slug,
            "letter": get_letter_from_term(slug)
        }

    # Plant entry: /apmtm/termino.php?l=3&t={slug}
    if "/apmtm/termino.php" in path:
        slug = query.get("t", [""])[0]
        return {
            "section": "plantas",
            "subsection": "por-nombre-botanico",
            "slug": slug
        }

    # Indigenous peoples: /mtpim/termino.php?v={p|r|d|m}&l=2&t={slug}
    if "/mtpim/termino.php" in path:
        slug = query.get("t", [""])[0]
        view = query.get("v", ["p"])[0]
        return {
            "section": "pueblos-indigenas",
            "slug": slug,
            "view": view
        }

    # Flora: /fmim/
    if "/fmim/" in path:
        return {
            "section": "flora-medicinal",
            "path": path
        }

    return None


def internal_url_to_markdown_link(url: str, from_section: str, from_subsection: str = None) -> str | None:
    """
    Convert an internal URL to a relative markdown link.

    Args:
        url: The internal URL to convert
        from_section: The section the current file is in (e.g., "plantas")
        from_subsection: The subsection if applicable (e.g., "por-nombre-botanico")

    Returns:
        Relative markdown path or None if not an internal URL
    """
    parsed = parse_internal_url(url)
    if not parsed:
        return None

    target_section = parsed["section"]

    # Build the target path
    if target_section == "diccionario":
        target_path = f"diccionario/{parsed['letter']}/{parsed['slug']}.md"
    elif target_section == "plantas":
        subsection = parsed.get("subsection", "por-nombre-botanico")
        target_path = f"plantas/{subsection}/{parsed['slug']}.md"
    elif target_section == "pueblos-indigenas":
        target_path = f"pueblos-indigenas/{parsed['slug']}.md"
    elif target_section == "flora-medicinal":
        target_path = f"flora-medicinal/index.md"
    else:
        return None

    # Calculate relative path from current location
    if from_section == "diccionario":
        # From: data/diccionario/{letter}/
        prefix = "../../"
    elif from_section == "plantas":
        # From: data/plantas/{subsection}/
        prefix = "../../"
    elif from_section == "pueblos-indigenas":
        # From: data/pueblos-indigenas/
        prefix = "../"
    elif from_section == "flora-medicinal":
        # From: data/flora-medicinal/
        prefix = "../"
    else:
        prefix = "../"

    return prefix + target_path


def convert_cross_references(html_content: str, soup, from_section: str, from_subsection: str = None) -> str:
    """
    Convert internal links in HTML/text to markdown links.

    Returns the text with cross-references converted to markdown links.
    """
    from bs4 import BeautifulSoup

    # Find all links
    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        text = link.get_text(strip=True)

        md_path = internal_url_to_markdown_link(href, from_section, from_subsection)
        if md_path:
            # Replace link with markdown link format
            link.replace_with(f"[{text}]({md_path})")

    return soup


def generate_frontmatter(metadata: dict) -> str:
    """Generate YAML frontmatter for a markdown file."""
    metadata["scraped_at"] = datetime.now().strftime("%Y-%m-%d")
    return "---\n" + yaml.dump(metadata, allow_unicode=True, default_flow_style=False) + "---\n\n"


def clean_text(text: str) -> str:
    """Clean up text by normalizing whitespace."""
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def html_to_markdown(soup_element) -> str:
    """
    Convert an HTML element to markdown text.
    Handles basic formatting: bold, italic, links, paragraphs.
    """
    if soup_element is None:
        return ""

    text = []

    for elem in soup_element.descendants:
        if hasattr(elem, 'name'):
            if elem.name == 'br':
                text.append('\n')
            elif elem.name == 'p':
                text.append('\n\n')
            elif elem.name in ('b', 'strong'):
                inner = elem.get_text()
                if inner.strip():
                    text.append(f"**{inner.strip()}**")
            elif elem.name in ('i', 'em'):
                inner = elem.get_text()
                if inner.strip():
                    text.append(f"*{inner.strip()}*")
            elif elem.name == 'a':
                # Links are handled by convert_cross_references
                pass
        elif hasattr(elem, 'string') and elem.string:
            text.append(str(elem.string))

    result = ''.join(text)
    # Clean up multiple newlines
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def get_image_filename(url: str, slug: str) -> str:
    """Generate a filename for a downloaded image."""
    parsed = urlparse(url)
    path = parsed.path
    ext = Path(path).suffix or ".jpg"
    return f"{slug}{ext}"


def get_image_path(section: str, filename: str) -> str:
    """Get the relative path for an image file."""
    return f"images/{section}/{filename}"


def ensure_directory(path: Path) -> None:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def save_markdown(content: str, filepath: Path) -> None:
    """Save markdown content to a file."""
    ensure_directory(filepath.parent)
    filepath.write_text(content, encoding="utf-8")


def load_progress(progress_file: Path) -> set:
    """Load the set of already-scraped URLs from progress file."""
    if progress_file.exists():
        return set(progress_file.read_text().strip().split("\n"))
    return set()


def save_progress(url: str, progress_file: Path) -> None:
    """Append a URL to the progress file."""
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(url + "\n")
