"""Parser for MTPIM - La Medicina Tradicional de los Pueblos Indígenas de México."""

import logging
import re
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ..utils import (
    BASE_URL,
    make_slug,
    generate_frontmatter,
    clean_text,
    internal_url_to_markdown_link,
    save_markdown,
    ensure_directory,
    get_image_filename,
)
from ..browser import BrowserManager

logger = logging.getLogger(__name__)


# Sub-sections for each indigenous people
PEOPLE_VIEWS = {
    "p": "La población",
    "r": "Los recursos humanos",
    "d": "Las demandas de atención",
    "m": "Descripción de demandas",
}


class PeoplesParser:
    """Parser for indigenous peoples entries."""

    def __init__(self, browser: BrowserManager, data_dir: Path):
        """
        Initialize parser.

        Args:
            browser: Browser manager instance
            data_dir: Base data directory
        """
        self.browser = browser
        self.data_dir = data_dir
        self.section_dir = data_dir / "pueblos-indigenas"
        self.images_dir = data_dir / "images" / "pueblos"

    async def get_all_peoples_urls(self) -> list[dict]:
        """
        Get all indigenous peoples entry URLs.

        Returns:
            List of dicts with 'url', 'text', 'slug' keys
        """
        entries = []
        letters = "abcdefghijklmnopqrstuvwxyz"

        for letter in letters:
            index_url = f"{BASE_URL}/mtpim/terminos-entrada.php?letra={letter}"
            logger.info(f"Scraping peoples index: letter {letter.upper()}")

            try:
                content = await self.browser.get_page_content(index_url)
                soup = BeautifulSoup(content, "lxml")

                for link in soup.find_all("a", href=True):
                    href = link.get("href", "")
                    if "termino.php" in href and "l=2" in href:
                        # Use index_url as base to correctly resolve relative URLs
                        full_url = urljoin(index_url, href)
                        text = clean_text(link.get_text())
                        if text:
                            # Extract slug from URL
                            slug_match = re.search(r"t=([^&]+)", href)
                            slug = slug_match.group(1) if slug_match else make_slug(text)

                            # Check if we already have this people
                            if not any(e["slug"] == slug for e in entries):
                                entries.append({
                                    "url": full_url,
                                    "text": text,
                                    "slug": slug
                                })

            except Exception as e:
                logger.error(f"Error scraping peoples letter {letter}: {e}")

        logger.info(f"Found {len(entries)} indigenous peoples")
        return entries

    async def parse_entry(self, base_url: str, slug: str, name: str) -> dict | None:
        """
        Parse all sections of an indigenous people entry.

        Args:
            base_url: Base URL of the entry
            slug: Slug identifier for the people
            name: Name of the people

        Returns:
            Parsed entry data or None if parsing failed
        """
        try:
            entry = {
                "name": name,
                "slug": slug,
                "url": base_url,
                "sections": {},
                "images": [],
            }

            # Fetch each sub-section
            for view_code, view_name in PEOPLE_VIEWS.items():
                section_url = f"{BASE_URL}/mtpim/termino.php?v={view_code}&l=2&t={slug}"
                logger.debug(f"Fetching section: {view_name}")

                try:
                    content = await self.browser.get_page_content(section_url)
                    soup = BeautifulSoup(content, "lxml")

                    main_content = soup.find("div", class_="contenido") or soup.find("td", class_="contenido") or soup.body

                    if main_content:
                        section_data = self._parse_section(main_content, view_name)
                        if section_data["content"]:
                            entry["sections"][view_name] = section_data["content"]

                        # Collect images
                        for img in main_content.find_all("img"):
                            src = img.get("src", "")
                            if src:
                                img_url = urljoin(BASE_URL, src)
                                if img_url not in [i["url"] for i in entry["images"]]:
                                    alt = img.get("alt", "")
                                    entry["images"].append({
                                        "url": img_url,
                                        "alt": alt
                                    })

                except Exception as e:
                    logger.error(f"Error fetching section {view_name}: {e}")

            return entry if entry["sections"] else None

        except Exception as e:
            logger.error(f"Error parsing people {name}: {e}")
            return None

    def _parse_section(self, content, section_name: str) -> dict:
        """Parse a single section's content."""
        result = {
            "content": "",
            "subsections": {}
        }

        paragraphs = []
        current_subsection = None
        current_subsection_content = []

        for elem in content.find_all(["p", "div", "h2", "h3", "h4", "strong", "b", "table", "ul", "ol"]):
            text = clean_text(elem.get_text())

            if not text:
                continue

            # Check for subsection headers (often bold text)
            if elem.name in ["strong", "b", "h3", "h4"] and len(text) < 100:
                # Save previous subsection
                if current_subsection and current_subsection_content:
                    result["subsections"][current_subsection] = "\n\n".join(current_subsection_content)

                current_subsection = text
                current_subsection_content = []
                continue

            # Handle lists
            if elem.name in ["ul", "ol"]:
                items = [f"- {clean_text(li.get_text())}" for li in elem.find_all("li")]
                text = "\n".join(items)

            # Handle tables
            if elem.name == "table":
                text = self._table_to_markdown(elem)

            # Add to appropriate place
            if current_subsection:
                current_subsection_content.append(text)
            else:
                paragraphs.append(text)

        # Save last subsection
        if current_subsection and current_subsection_content:
            result["subsections"][current_subsection] = "\n\n".join(current_subsection_content)

        # Combine main content
        result["content"] = "\n\n".join(paragraphs)

        # Add subsections to content
        for sub_name, sub_content in result["subsections"].items():
            result["content"] += f"\n\n### {sub_name}\n\n{sub_content}"

        return result

    def _table_to_markdown(self, table) -> str:
        """Convert an HTML table to markdown format."""
        rows = []
        for tr in table.find_all("tr"):
            cells = [clean_text(td.get_text()) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append("| " + " | ".join(cells) + " |")

        if not rows:
            return ""

        if len(rows) > 1:
            num_cols = rows[0].count("|") - 1
            separator = "| " + " | ".join(["---"] * num_cols) + " |"
            rows.insert(1, separator)

        return "\n".join(rows)

    def _convert_cross_references(self, text: str) -> str:
        """Convert cross-references in text to markdown links."""
        # This is a simplified version; in practice, we'd need to track the actual links
        return text

    def generate_markdown(self, entry: dict, image_paths: list[str] = None) -> str:
        """
        Generate markdown content for a people entry.

        Args:
            entry: Parsed entry data
            image_paths: List of relative paths to downloaded images

        Returns:
            Markdown content
        """
        # Frontmatter
        frontmatter = generate_frontmatter({
            "title": entry["name"],
            "source": entry["url"],
        })

        lines = [frontmatter]

        # Title
        lines.append(f"# {entry['name']}\n")

        # Images at the top
        if image_paths:
            for i, img_path in enumerate(image_paths):
                alt = entry["images"][i].get("alt", entry["name"]) if i < len(entry.get("images", [])) else entry["name"]
                lines.append(f"![{alt}]({img_path})")
            lines.append("")

        # Sections
        for section_name in PEOPLE_VIEWS.values():
            if section_name in entry["sections"]:
                content = entry["sections"][section_name]
                if content.strip():
                    lines.append(f"## {section_name}\n")
                    lines.append(content)
                    lines.append("")

        return "\n".join(lines)

    async def scrape_entry(self, url: str, slug: str, name: str) -> Path | None:
        """
        Scrape and save a single indigenous people entry.

        Args:
            url: URL of the entry
            slug: Slug identifier
            name: Name of the people

        Returns:
            Path to saved file or None if failed
        """
        entry = await self.parse_entry(url, slug, name)
        if not entry:
            return None

        ensure_directory(self.section_dir)
        filepath = self.section_dir / f"{slug}.md"

        # Download images
        image_paths = []
        for i, img in enumerate(entry.get("images", [])):
            img_filename = get_image_filename(img["url"], f"{slug}-{i+1}")
            img_save_path = self.images_dir / img_filename
            if await self.browser.download_image(img["url"], img_save_path):
                image_paths.append(f"../images/pueblos/{img_filename}")

        # Generate and save markdown
        markdown = self.generate_markdown(entry, image_paths)
        save_markdown(markdown, filepath)

        logger.info(f"Saved: {filepath}")
        return filepath

    async def scrape_all(self, progress_file: Path = None) -> int:
        """
        Scrape all indigenous peoples entries.

        Args:
            progress_file: Optional file to track progress

        Returns:
            Number of entries scraped
        """
        from ..utils import load_progress, save_progress

        entries = await self.get_all_peoples_urls()
        scraped = set()

        if progress_file:
            scraped = load_progress(progress_file)

        count = 0
        total = len(entries)

        for i, entry in enumerate(entries):
            url = entry["url"]

            if url in scraped:
                logger.debug(f"Skipping already scraped: {url}")
                continue

            logger.info(f"[{i+1}/{total}] Scraping: {entry['text']}")

            try:
                filepath = await self.scrape_entry(url, entry["slug"], entry["text"])
                if filepath and progress_file:
                    save_progress(url, progress_file)
                    count += 1
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")

        logger.info(f"Scraped {count} new peoples entries")
        return count
