"""Parser for DEMTM - Diccionario Enciclopédico de la Medicina Tradicional Mexicana."""

import logging
import re
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ..utils import (
    BASE_URL,
    make_slug,
    get_letter_from_term,
    generate_frontmatter,
    clean_text,
    internal_url_to_markdown_link,
    save_markdown,
    ensure_directory,
)
from ..browser import BrowserManager

logger = logging.getLogger(__name__)


class DictionaryParser:
    """Parser for dictionary entries."""

    def __init__(self, browser: BrowserManager, data_dir: Path):
        """
        Initialize parser.

        Args:
            browser: Browser manager instance
            data_dir: Base data directory
        """
        self.browser = browser
        self.data_dir = data_dir
        self.section_dir = data_dir / "diccionario"

    async def get_all_entry_urls(self) -> list[dict]:
        """
        Get all dictionary entry URLs by scraping the A-Z index.

        Returns:
            List of dicts with 'url', 'text' keys
        """
        entries = []
        letters = "abcdefghijklmnopqrstuvwxyz"

        for letter in letters:
            index_url = f"{BASE_URL}/demtm/terminos-entrada.php?letra={letter}"
            logger.info(f"Scraping dictionary index: letter {letter.upper()}")

            try:
                content = await self.browser.get_page_content(index_url)
                soup = BeautifulSoup(content, "lxml")

                # Find entry links - they typically link to termino.php
                for link in soup.find_all("a", href=True):
                    href = link.get("href", "")
                    if "termino.php" in href and "l=1" in href:
                        # Use index_url as base to correctly resolve relative URLs
                        full_url = urljoin(index_url, href)
                        text = clean_text(link.get_text())
                        if text:
                            entries.append({
                                "url": full_url,
                                "text": text,
                                "letter": letter
                            })

            except Exception as e:
                logger.error(f"Error scraping letter {letter}: {e}")

        logger.info(f"Found {len(entries)} dictionary entries")
        return entries

    async def parse_entry(self, url: str) -> dict | None:
        """
        Parse a single dictionary entry.

        Args:
            url: URL of the entry

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

            # Find the main content area - look for article tags (the site uses semantic HTML5)
            articles = soup.find_all("article")
            main_content = None
            title = None

            # The content is typically in the last article with paragraphs
            for article in articles:
                paragraphs = article.find_all("p")
                if paragraphs:
                    main_content = article
                    # Title is often in a div before the paragraphs
                    title_div = article.find("div", recursive=False)
                    if title_div:
                        # Look for title text (not the header image)
                        for div in article.find_all("div", recursive=False):
                            div_text = clean_text(div.get_text())
                            # Skip navigation-like text and long text
                            if div_text and len(div_text) < 100 and ">>" not in div_text:
                                if not div.find("img"):  # Skip divs with images
                                    title = div_text
                                    break

            if not main_content:
                main_content = soup.body

            # Try to get title from page title if not found
            if not title:
                page_title = soup.find("title")
                if page_title:
                    title_text = clean_text(page_title.get_text())
                    # Extract term name from ":: Abeja - Términos - DEMTM ::"
                    match = re.match(r"::\s*(.+?)\s*-\s*Términos", title_text)
                    if match:
                        title = match.group(1)
                    else:
                        title = title_text

            if not title or "Not Found" in title:
                logger.warning(f"No valid title found for {url}")
                return None

            # Extract content paragraphs
            description_parts = []
            bibliography = []
            in_bibliography = False

            for p in main_content.find_all("p"):
                text = clean_text(p.get_text())
                if not text:
                    continue

                # Check if this is bibliography (numbered references pattern)
                if re.match(r'^\(\d+\)', text) or "Índice de Autores" in text:
                    in_bibliography = True

                if in_bibliography:
                    # Parse bibliography entries
                    if text != "Índice de Autores":
                        bibliography.append(text)
                else:
                    # Convert links to markdown format
                    p_html = str(p)
                    for link in p.find_all("a", href=True):
                        href = link.get("href", "")
                        link_text = clean_text(link.get_text())
                        md_link = internal_url_to_markdown_link(href, "diccionario")
                        if md_link and link_text:
                            # Create markdown link
                            p_html = p_html.replace(str(link), f"[{link_text}]({md_link})")

                    # Clean up HTML and extract text with markdown links preserved
                    from bs4 import BeautifulSoup as BS
                    p_soup = BS(p_html, "lxml")
                    # The text now has markdown links embedded
                    desc_text = clean_text(p_soup.get_text())
                    # But we need to preserve the markdown links, so use the html
                    # Just extract text and handle links manually
                    desc_with_links = text
                    for link in p.find_all("a", href=True):
                        href = link.get("href", "")
                        link_text = clean_text(link.get_text())
                        md_link = internal_url_to_markdown_link(href, "diccionario")
                        if md_link and link_text:
                            desc_with_links = desc_with_links.replace(
                                link_text,
                                f"[{link_text}]({md_link})",
                                1  # Only first occurrence
                            )
                    description_parts.append(desc_with_links)

            return {
                "title": title,
                "url": url,
                "synonyms": [],
                "indigenous_terms": [],
                "description": "\n\n".join(description_parts),
                "bibliography": bibliography,
            }

        except Exception as e:
            logger.error(f"Error parsing entry {url}: {e}")
            return None

    def generate_markdown(self, entry: dict) -> str:
        """
        Generate markdown content for a dictionary entry.

        Args:
            entry: Parsed entry data

        Returns:
            Markdown content
        """
        # Frontmatter
        frontmatter = generate_frontmatter({
            "title": entry["title"],
            "source": entry["url"],
        })

        lines = [frontmatter]

        # Title
        lines.append(f"# {entry['title']}\n")

        # Synonyms
        if entry["synonyms"]:
            lines.append("## Sinonimia\n")
            for syn in entry["synonyms"]:
                lines.append(f"- {syn}")
            lines.append("")

        # Indigenous terms
        if entry["indigenous_terms"]:
            lines.append("## Términos en lenguas indígenas\n")
            for term in entry["indigenous_terms"]:
                lines.append(f"- {term}")
            lines.append("")

        # Description
        if entry["description"]:
            lines.append("## Descripción\n")
            lines.append(entry["description"])
            lines.append("")

        # Bibliography
        if entry["bibliography"]:
            lines.append("## Bibliografía\n")
            for bib in entry["bibliography"]:
                lines.append(f"- {bib}")
            lines.append("")

        return "\n".join(lines)

    async def scrape_entry(self, url: str, text: str = None) -> Path | None:
        """
        Scrape and save a single dictionary entry.

        Args:
            url: URL of the entry
            text: Optional text/name of the entry

        Returns:
            Path to saved file or None if failed
        """
        entry = await self.parse_entry(url)
        if not entry:
            return None

        # Determine save path
        slug = make_slug(entry["title"])
        letter = get_letter_from_term(entry["title"])
        letter_dir = self.section_dir / letter
        ensure_directory(letter_dir)

        filepath = letter_dir / f"{slug}.md"

        # Generate and save markdown
        markdown = self.generate_markdown(entry)
        save_markdown(markdown, filepath)

        logger.info(f"Saved: {filepath}")
        return filepath

    async def scrape_all(self, progress_file: Path = None) -> int:
        """
        Scrape all dictionary entries.

        Args:
            progress_file: Optional file to track progress

        Returns:
            Number of entries scraped
        """
        from ..utils import load_progress, save_progress

        entries = await self.get_all_entry_urls()
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

        logger.info(f"Scraped {count} new dictionary entries")
        return count
