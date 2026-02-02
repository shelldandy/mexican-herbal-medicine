"""Parser for APMTM - Atlas de las Plantas de la Medicina Tradicional Mexicana."""

import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs

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


# Section headers that may appear in plant entries
PLANT_SECTIONS = [
    "Sinonimia botánica",
    "Sinonimia popular",
    "Botánica y ecología",
    "Etnobotánica y antropología",
    "Historia",
    "Química",
    "Farmacología",
    "Principios activos",
    "Toxicidad",
    "Comentarios",
    "Herbarios",
    "Literatura",
]


class PlantsParser:
    """Parser for plant monographs."""

    def __init__(self, browser: BrowserManager, data_dir: Path):
        """
        Initialize parser.

        Args:
            browser: Browser manager instance
            data_dir: Base data directory
        """
        self.browser = browser
        self.data_dir = data_dir
        self.section_dir = data_dir / "plantas"
        self.images_dir = data_dir / "images" / "plantas"

    async def get_all_plant_urls(self, by_botanical: bool = True) -> list[dict]:
        """
        Get all plant entry URLs.

        Args:
            by_botanical: If True, get by botanical name; if False, by popular name

        Returns:
            List of dicts with 'url', 'text', 'subsection' keys
        """
        entries = []
        option = "nb" if by_botanical else "np"
        subsection = "por-nombre-botanico" if by_botanical else "por-nombre-popular"
        letters = "abcdefghijklmnopqrstuvwxyz"

        for letter in letters:
            index_url = f"{BASE_URL}/apmtm/terminos-entrada.php?letra={letter}&opcion={option}"
            logger.info(f"Scraping plants index ({subsection}): letter {letter.upper()}")

            try:
                content = await self.browser.get_page_content(index_url)
                soup = BeautifulSoup(content, "lxml")

                for link in soup.find_all("a", href=True):
                    href = link.get("href", "")
                    if "termino.php" in href and "l=3" in href:
                        # Use index_url as base to correctly resolve relative URLs
                        full_url = urljoin(index_url, href)
                        text = clean_text(link.get_text())
                        if text:
                            entries.append({
                                "url": full_url,
                                "text": text,
                                "subsection": subsection
                            })

            except Exception as e:
                logger.error(f"Error scraping plants letter {letter}: {e}")

        logger.info(f"Found {len(entries)} plant entries ({subsection})")
        return entries

    async def get_fungi_urls(self, by_botanical: bool = True) -> list[dict]:
        """
        Get all fungi entry URLs.

        Args:
            by_botanical: If True, get by botanical name; if False, by popular name

        Returns:
            List of dicts with 'url', 'text', 'subsection' keys
        """
        entries = []
        option = "hnb" if by_botanical else "hnp"
        letters = "abcdefghijklmnopqrstuvwxyz"

        for letter in letters:
            index_url = f"{BASE_URL}/apmtm/terminos-entrada.php?letra={letter}&opcion={option}"
            logger.info(f"Scraping fungi index: letter {letter.upper()}")

            try:
                content = await self.browser.get_page_content(index_url)
                soup = BeautifulSoup(content, "lxml")

                for link in soup.find_all("a", href=True):
                    href = link.get("href", "")
                    if "termino.php" in href:
                        # Use index_url as base to correctly resolve relative URLs
                        full_url = urljoin(index_url, href)
                        text = clean_text(link.get_text())
                        if text:
                            entries.append({
                                "url": full_url,
                                "text": text,
                                "subsection": "hongos"
                            })

            except Exception as e:
                logger.error(f"Error scraping fungi letter {letter}: {e}")

        logger.info(f"Found {len(entries)} fungi entries")
        return entries

    async def parse_entry(self, url: str, subsection: str = "por-nombre-botanico") -> dict | None:
        """
        Parse a single plant entry.

        Args:
            url: URL of the entry
            subsection: The subsection (por-nombre-botanico, por-nombre-popular, hongos)

        Returns:
            Parsed entry data or None if parsing failed
        """
        try:
            content = await self.browser.get_page_content(url)
            soup = BeautifulSoup(content, "lxml")

            # Check for 404/not found
            if "Not Found" in soup.get_text() or "404" in soup.get_text():
                logger.warning(f"Page not found: {url}")
                return None

            # Extract basic info
            entry = {
                "url": url,
                "subsection": subsection,
                "title": None,
                "botanical_name": None,
                "family": None,
                "image_url": None,
                "image_credit": None,
                "sections": {},
            }

            # Find the main content article (usually the last one with substantial content)
            articles = soup.find_all("article")
            main_content = None
            for article in articles:
                if article.find("p"):
                    main_content = article
                    break

            if not main_content:
                main_content = soup.body

            # Try to get title from page title ":: Sábila - Términos - APMTM ::"
            page_title = soup.find("title")
            if page_title:
                title_text = clean_text(page_title.get_text())
                match = re.match(r"::\s*(.+?)\s*-\s*Términos", title_text)
                if match:
                    entry["title"] = match.group(1)

            # Look for botanical name and family in the structure
            # Pattern: "Aloe vera L. — Liliaceae" (may have em-dash or regular dash)
            full_text = main_content.get_text()
            # Try em-dash first, then regular dash
            for dash in ["—", "-", "–"]:
                if dash in full_text:
                    for div in main_content.find_all("div", recursive=True):
                        text = clean_text(div.get_text())
                        if dash in text and len(text) < 200:
                            parts = text.split(dash)
                            if len(parts) >= 2:
                                botanical = parts[0].strip()
                                family = parts[1].strip()
                                # Validate it looks like botanical name (has latin species format)
                                if re.match(r'^[A-Z][a-z]+\s+[a-z]+', botanical):
                                    entry["botanical_name"] = botanical
                                    entry["family"] = family
                                    break
                    if entry["botanical_name"]:
                        break

            # If still no botanical name, try the link text for the current page
            if not entry["botanical_name"]:
                for link in main_content.find_all("a", href=True):
                    href = link.get("href", "")
                    if "l=3" in href and "t=" in href:
                        botanical = clean_text(link.get_text())
                        if re.match(r'^[A-Z][a-z]+\s+[a-z]+', botanical):
                            entry["botanical_name"] = botanical
                            break

            # Fallback family extraction from content patterns
            if not entry["family"]:
                # Try pattern: **Familia:** FamilyName
                family_match = re.search(r'\*\*Familia:\*\*\s*([^\n*]+)', full_text)
                if family_match:
                    entry["family"] = family_match.group(1).strip()

                # Try pattern: "Familia: FamilyName" or "Familia FamilyName"
                if not entry["family"]:
                    family_match = re.search(r'Familia[:\s]+([A-Z][a-z]+(?:aceae|ae|eae))\b', full_text)
                    if family_match:
                        entry["family"] = family_match.group(1).strip()

                # Try extracting from section content if present
                if not entry["family"] and "Sinonimia botánica" in entry["sections"]:
                    section_text = entry["sections"]["Sinonimia botánica"]
                    family_match = re.search(r'([A-Z][a-z]+(?:aceae|ae|eae))\b', section_text)
                    if family_match:
                        entry["family"] = family_match.group(1).strip()

            # Find image - look for links to images in /imagenes/
            for link in main_content.find_all("a", href=True):
                href = link.get("href", "")
                if "/imagenes/" in href and any(ext in href.lower() for ext in [".jpg", ".jpeg", ".png", ".gif"]):
                    entry["image_url"] = urljoin(url, href)
                    # Look for image credit in nearby cell
                    cell = link.find_parent("td")
                    if cell:
                        next_row = cell.find_parent("tr")
                        if next_row:
                            next_row = next_row.find_next_sibling("tr")
                            if next_row:
                                credit = clean_text(next_row.get_text())
                                if "imagen" in credit.lower() or "proporcionada" in credit.lower():
                                    entry["image_credit"] = credit
                    break

            # Parse sections by looking for section headers (in generic/div elements)
            current_section = None
            current_content = []

            for elem in main_content.find_all(["p", "div"], recursive=True):
                text = clean_text(elem.get_text())

                if not text:
                    continue

                # Check if this is a section header (ends with period and matches known sections)
                is_section = False
                for section_name in PLANT_SECTIONS:
                    # Section headers often end with period like "Sinonimia botánica."
                    if text.rstrip(".").lower() == section_name.lower() or \
                       text.lower() == section_name.lower() + ".":
                        # Save previous section
                        if current_section and current_content:
                            entry["sections"][current_section] = self._process_section_content(
                                current_content, main_content, subsection
                            )
                        current_section = section_name
                        current_content = []
                        is_section = True
                        break

                if not is_section and current_section and elem.name == "p":
                    # Process paragraph with cross-references
                    para_text = self._process_paragraph(elem, subsection)
                    if para_text:
                        current_content.append(para_text)

            # Save last section
            if current_section and current_content:
                entry["sections"][current_section] = self._process_section_content(
                    current_content, main_content, subsection
                )

            # If no structured sections found, get all text as description
            if not entry["sections"]:
                all_text = clean_text(main_content.get_text())
                if all_text:
                    entry["sections"]["Descripción"] = all_text

            return entry

        except Exception as e:
            logger.error(f"Error parsing plant {url}: {e}")
            return None

    def _process_paragraph(self, p_elem, subsection: str) -> str:
        """Process a paragraph element and convert cross-references to markdown links."""
        text = clean_text(p_elem.get_text())

        # Convert each link in the paragraph
        for link in p_elem.find_all("a", href=True):
            href = link.get("href", "")
            link_text = clean_text(link.get_text())
            md_link = internal_url_to_markdown_link(href, "plantas", subsection)
            if md_link and link_text:
                # Replace the link text with markdown link
                text = text.replace(link_text, f"[{link_text}]({md_link})", 1)

        return text

    def _process_section_content(self, content: list, main_content, subsection: str) -> str:
        """Process section content - content already has cross-references converted."""
        return "\n\n".join(content)

    def _table_to_markdown(self, table) -> str:
        """Convert an HTML table to markdown format."""
        rows = []
        for tr in table.find_all("tr"):
            cells = [clean_text(td.get_text()) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append("| " + " | ".join(cells) + " |")

        if not rows:
            return ""

        # Add header separator after first row
        if len(rows) > 1:
            num_cols = rows[0].count("|") - 1
            separator = "| " + " | ".join(["---"] * num_cols) + " |"
            rows.insert(1, separator)

        return "\n".join(rows)

    def generate_markdown(self, entry: dict, image_path: str = None) -> str:
        """
        Generate markdown content for a plant entry.

        Args:
            entry: Parsed entry data
            image_path: Relative path to downloaded image

        Returns:
            Markdown content
        """
        # Frontmatter
        frontmatter_data = {
            "title": entry["title"],
            "source": entry["url"],
        }
        if entry["botanical_name"]:
            frontmatter_data["botanical_name"] = entry["botanical_name"]
        if entry["family"]:
            frontmatter_data["family"] = entry["family"]

        frontmatter = generate_frontmatter(frontmatter_data)

        lines = [frontmatter]

        # Title
        lines.append(f"# {entry['title']}\n")

        # Basic info
        if entry["botanical_name"]:
            lines.append(f"**Nombre botánico:** {entry['botanical_name']}")
        if entry["family"]:
            lines.append(f"**Familia:** {entry['family']}")
        lines.append("")

        # Image
        if image_path:
            lines.append(f"![{entry['title']}]({image_path})")
            if entry.get("image_credit"):
                lines.append(f"_{entry['image_credit']}_")
            lines.append("")

        # Sections
        for section_name in PLANT_SECTIONS:
            if section_name in entry["sections"]:
                content = entry["sections"][section_name]
                if content.strip():
                    lines.append(f"## {section_name}\n")
                    lines.append(content)
                    lines.append("")

        # Any other sections not in the standard list
        for section_name, content in entry["sections"].items():
            if section_name not in PLANT_SECTIONS and content.strip():
                lines.append(f"## {section_name}\n")
                lines.append(content)
                lines.append("")

        return "\n".join(lines)

    async def scrape_entry(self, url: str, subsection: str = "por-nombre-botanico") -> Path | None:
        """
        Scrape and save a single plant entry.

        Args:
            url: URL of the entry
            subsection: The subsection to save to

        Returns:
            Path to saved file or None if failed
        """
        entry = await self.parse_entry(url, subsection)
        if not entry:
            return None

        # Determine save path
        slug = make_slug(entry["title"])
        output_dir = self.section_dir / subsection
        ensure_directory(output_dir)

        filepath = output_dir / f"{slug}.md"

        # Download image if present
        image_path = None
        if entry["image_url"]:
            img_filename = get_image_filename(entry["image_url"], slug)
            img_save_path = self.images_dir / img_filename
            if await self.browser.download_image(entry["image_url"], img_save_path):
                # Relative path from markdown file to image
                image_path = f"../../images/plantas/{img_filename}"

        # Generate and save markdown
        markdown = self.generate_markdown(entry, image_path)
        save_markdown(markdown, filepath)

        logger.info(f"Saved: {filepath}")
        return filepath

    async def scrape_all(
        self,
        progress_file: Path = None,
        include_fungi: bool = True,
        include_popular_names: bool = True,
    ) -> int:
        """
        Scrape all plant entries.

        Args:
            progress_file: Optional file to track progress
            include_fungi: Whether to also scrape fungi
            include_popular_names: Whether to also scrape plants by popular name

        Returns:
            Number of entries scraped
        """
        from ..utils import load_progress, save_progress

        # Get all URLs by botanical name
        entries = await self.get_all_plant_urls(by_botanical=True)

        # Also get plants by popular name
        if include_popular_names:
            popular_entries = await self.get_all_plant_urls(by_botanical=False)
            entries.extend(popular_entries)

        if include_fungi:
            fungi = await self.get_fungi_urls(by_botanical=True)
            entries.extend(fungi)

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
                filepath = await self.scrape_entry(url, entry["subsection"])
                if filepath and progress_file:
                    save_progress(url, progress_file)
                    count += 1
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")

        logger.info(f"Scraped {count} new plant entries")
        return count
