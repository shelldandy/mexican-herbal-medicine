"""Parser for FMIM - Flora Medicinal Indígena de México."""

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


class FloraParser:
    """Parser for Flora Medicinal Indígena de México section."""

    def __init__(self, browser: BrowserManager, data_dir: Path):
        """
        Initialize parser.

        Args:
            browser: Browser manager instance
            data_dir: Base data directory
        """
        self.browser = browser
        self.data_dir = data_dir
        self.section_dir = data_dir / "flora-medicinal"
        self.images_dir = data_dir / "images" / "flora"

    async def get_all_flora_urls(self) -> list[dict]:
        """
        Get all flora section URLs.

        The flora monographs are listed on the introduction page (/fmim/introduccion.html).
        Each flora entry links to a page like /fmim/flora_cochimi.html, /fmim/flora_maya.html, etc.

        Returns:
            List of dicts with 'url', 'text' keys
        """
        entries = []
        index_url = f"{BASE_URL}/fmim/"
        intro_url = f"{BASE_URL}/fmim/introduccion.html"

        # Known flora monograph patterns - these are the expected indigenous flora studies
        known_monographs = [
            ("flora_cochimi", "Flora Cochimí"),
            ("flora_cucapa", "Flora Cucapá"),
            ("flora_kiliwa", "Flora Kiliwa"),
            ("flora_kumiai", "Flora Kumiai"),
            ("flora_paipai", "Flora Pai-pai"),
            ("flora_maya", "Flora Maya"),
            ("flora_nahua", "Flora Nahua"),
            ("flora_zapoteca", "Flora Zapoteca"),
            ("flora_mixe", "Flora Mixe"),
            ("flora_mixteca", "Flora Mixteca"),
            ("flora_mazateca", "Flora Mazateca"),
            ("flora_chinanteca", "Flora Chinanteca"),
        ]

        logger.info("Scraping flora index and introduction page")

        try:
            # First, add the main index page
            entries.append({
                "url": index_url,
                "text": "Introducción"
            })

            # Try known monograph URLs directly
            for monograph_slug, monograph_name in known_monographs:
                monograph_url = f"{BASE_URL}/fmim/{monograph_slug}.html"
                entries.append({
                    "url": monograph_url,
                    "text": monograph_name
                })

            # Parse the introduction page to find additional flora monograph links
            content = await self.browser.get_page_content(intro_url)
            soup = BeautifulSoup(content, "lxml")

            # Find all links to flora pages
            # Flora links follow patterns like: /fmim/flora_cochimi.html, /fmim/flora_maya.html
            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                text = clean_text(link.get_text())

                # Skip empty or navigation links
                if not text or len(text) < 3:
                    continue

                # Match flora page links (flora_*.html or direct /fmim/ links)
                if "flora" in href.lower() or "/fmim/" in href:
                    # Resolve relative URLs
                    full_url = urljoin(intro_url, href)

                    # Skip the intro page itself and index
                    if "introduccion" in full_url.lower() or full_url == index_url:
                        continue

                    # Avoid duplicates
                    if full_url not in [e["url"] for e in entries]:
                        entries.append({
                            "url": full_url,
                            "text": text
                        })

            # Also try the main index page for any additional links
            index_content = await self.browser.get_page_content(index_url)
            index_soup = BeautifulSoup(index_content, "lxml")

            for link in index_soup.find_all("a", href=True):
                href = link.get("href", "")
                text = clean_text(link.get_text())

                if not text or len(text) < 3:
                    continue

                # Look for links containing "Flora" in the text (Flora Maya, Flora Kiliwa, etc.)
                if "Flora" in text or "flora" in href.lower():
                    full_url = urljoin(index_url, href)

                    # Skip intro and index
                    if "introduccion" in full_url.lower() or full_url == index_url:
                        continue

                    if full_url not in [e["url"] for e in entries]:
                        entries.append({
                            "url": full_url,
                            "text": text
                        })

        except Exception as e:
            logger.error(f"Error scraping flora index: {e}")

        logger.info(f"Found {len(entries)} flora pages")
        return entries

    async def parse_entry(self, url: str, title: str = None) -> dict | None:
        """
        Parse a single flora page.

        Args:
            url: URL of the page
            title: Optional title override

        Returns:
            Parsed entry data or None if parsing failed
        """
        try:
            content = await self.browser.get_page_content(url)
            soup = BeautifulSoup(content, "lxml")

            main_content = soup.find("div", class_="contenido") or soup.find("td", class_="contenido") or soup.body

            if not main_content:
                logger.warning(f"No content found for {url}")
                return None

            entry = {
                "url": url,
                "title": title,
                "content": "",
                "sections": {},
                "images": [],
            }

            # Get title from page if not provided
            if not entry["title"]:
                title_elem = main_content.find(["h1", "h2", "h3"])
                if title_elem:
                    entry["title"] = clean_text(title_elem.get_text())
                else:
                    page_title = soup.find("title")
                    if page_title:
                        entry["title"] = clean_text(page_title.get_text())

            if not entry["title"]:
                entry["title"] = "Flora Medicinal Indígena de México"

            # Collect images
            for img in main_content.find_all("img"):
                src = img.get("src", "")
                if src:
                    img_url = urljoin(BASE_URL, src)
                    alt = img.get("alt", "")
                    entry["images"].append({
                        "url": img_url,
                        "alt": alt
                    })

            # Parse content
            current_section = None
            current_content = []
            general_content = []

            for elem in main_content.find_all(["p", "div", "h2", "h3", "h4", "ul", "ol", "table"]):
                text = clean_text(elem.get_text())

                if not text:
                    continue

                # Check for section headers
                if elem.name in ["h2", "h3", "h4"]:
                    # Save previous section
                    if current_section and current_content:
                        entry["sections"][current_section] = "\n\n".join(current_content)

                    current_section = text
                    current_content = []
                    continue

                # Handle lists
                if elem.name in ["ul", "ol"]:
                    items = [f"- {clean_text(li.get_text())}" for li in elem.find_all("li")]
                    text = "\n".join(items)

                # Handle tables
                if elem.name == "table":
                    text = self._table_to_markdown(elem)

                # Add to appropriate place
                if current_section:
                    current_content.append(text)
                else:
                    general_content.append(text)

            # Save last section
            if current_section and current_content:
                entry["sections"][current_section] = "\n\n".join(current_content)

            # Set main content
            entry["content"] = "\n\n".join(general_content)

            return entry

        except Exception as e:
            logger.error(f"Error parsing flora page {url}: {e}")
            return None

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

    def generate_markdown(self, entry: dict, image_paths: list[str] = None) -> str:
        """
        Generate markdown content for a flora page.

        Args:
            entry: Parsed entry data
            image_paths: List of relative paths to downloaded images

        Returns:
            Markdown content
        """
        frontmatter = generate_frontmatter({
            "title": entry["title"],
            "source": entry["url"],
        })

        lines = [frontmatter]

        # Title
        lines.append(f"# {entry['title']}\n")

        # Images
        if image_paths:
            for i, img_path in enumerate(image_paths):
                alt = entry["images"][i].get("alt", entry["title"]) if i < len(entry.get("images", [])) else entry["title"]
                lines.append(f"![{alt}]({img_path})")
            lines.append("")

        # Main content
        if entry["content"]:
            lines.append(entry["content"])
            lines.append("")

        # Sections
        for section_name, section_content in entry["sections"].items():
            if section_content.strip():
                lines.append(f"## {section_name}\n")
                lines.append(section_content)
                lines.append("")

        return "\n".join(lines)

    async def scrape_entry(self, url: str, title: str) -> Path | None:
        """
        Scrape and save a single flora page.

        Args:
            url: URL of the page
            title: Title of the page

        Returns:
            Path to saved file or None if failed
        """
        entry = await self.parse_entry(url, title)
        if not entry:
            return None

        ensure_directory(self.section_dir)
        ensure_directory(self.images_dir)

        # Determine filename
        slug = make_slug(entry["title"])
        if "introducción" in entry["title"].lower() or "index" in url:
            filename = "index.md"
        else:
            filename = f"{slug}.md"

        filepath = self.section_dir / filename

        # Download images
        image_paths = []
        for i, img in enumerate(entry.get("images", [])):
            img_filename = get_image_filename(img["url"], f"{slug}-{i+1}")
            img_save_path = self.images_dir / img_filename
            if await self.browser.download_image(img["url"], img_save_path):
                image_paths.append(f"../images/flora/{img_filename}")

        # Generate and save markdown
        markdown = self.generate_markdown(entry, image_paths)
        save_markdown(markdown, filepath)

        logger.info(f"Saved: {filepath}")
        return filepath

    async def scrape_all(self, progress_file: Path = None) -> int:
        """
        Scrape all flora pages.

        Args:
            progress_file: Optional file to track progress

        Returns:
            Number of pages scraped
        """
        from ..utils import load_progress, save_progress

        entries = await self.get_all_flora_urls()
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
                filepath = await self.scrape_entry(url, entry["text"])
                if filepath and progress_file:
                    save_progress(url, progress_file)
                    count += 1
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")

        logger.info(f"Scraped {count} new flora pages")
        return count
