#!/usr/bin/env python3
"""
UNAM Traditional Mexican Medicine Digital Library Scraper

Scrapes data from http://www.medicinatradicionalmexicana.unam.mx/
and stores it as organized markdown files.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .browser import BrowserManager
from .parsers.dictionary import DictionaryParser
from .parsers.plants import PlantsParser
from .parsers.peoples import PeoplesParser
from .parsers.flora import FloraParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scraper.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


async def scrape_section(
    section: str,
    browser: BrowserManager,
    data_dir: Path,
    progress_dir: Path,
    letter: str = None,
) -> int:
    """
    Scrape a specific section.

    Args:
        section: Section to scrape (dictionary, plants, peoples, flora, all)
        browser: Browser manager instance
        data_dir: Base data directory
        progress_dir: Directory for progress files
        letter: Optional letter to filter (for testing)

    Returns:
        Number of entries scraped
    """
    count = 0

    if section in ("dictionary", "all"):
        logger.info("=" * 50)
        logger.info("Scraping Dictionary (DEMTM)")
        logger.info("=" * 50)
        parser = DictionaryParser(browser, data_dir)
        progress_file = progress_dir / "dictionary_progress.txt"

        if letter:
            # Scrape only specific letter for testing
            entries = await parser.get_all_entry_urls()
            entries = [e for e in entries if e["letter"] == letter.lower()]
            for entry in entries:
                await parser.scrape_entry(entry["url"], entry["text"])
            count += len(entries)
        else:
            count += await parser.scrape_all(progress_file)

    if section in ("plants", "all"):
        logger.info("=" * 50)
        logger.info("Scraping Plants (APMTM)")
        logger.info("=" * 50)
        parser = PlantsParser(browser, data_dir)
        progress_file = progress_dir / "plants_progress.txt"

        if letter:
            # Get plants by botanical name
            entries = await parser.get_all_plant_urls(by_botanical=True)
            # Also get plants by popular name
            popular_entries = await parser.get_all_plant_urls(by_botanical=False)
            entries.extend(popular_entries)

            entries = [e for e in entries if e["text"].lower().startswith(letter.lower())]
            for entry in entries:
                await parser.scrape_entry(entry["url"], entry["subsection"])
            count += len(entries)
        else:
            count += await parser.scrape_all(progress_file)

    if section in ("peoples", "all"):
        logger.info("=" * 50)
        logger.info("Scraping Indigenous Peoples (MTPIM)")
        logger.info("=" * 50)
        parser = PeoplesParser(browser, data_dir)
        progress_file = progress_dir / "peoples_progress.txt"

        if letter:
            entries = await parser.get_all_peoples_urls()
            entries = [e for e in entries if e["text"].lower().startswith(letter.lower())]
            for entry in entries:
                await parser.scrape_entry(entry["url"], entry["slug"], entry["text"])
            count += len(entries)
        else:
            count += await parser.scrape_all(progress_file)

    if section in ("flora", "all"):
        logger.info("=" * 50)
        logger.info("Scraping Flora Medicinal (FMIM)")
        logger.info("=" * 50)
        parser = FloraParser(browser, data_dir)
        progress_file = progress_dir / "flora_progress.txt"
        count += await parser.scrape_all(progress_file)

    return count


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    data_dir = Path(args.output)
    progress_dir = data_dir / ".progress"
    progress_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting scraper")
    logger.info(f"Output directory: {data_dir}")
    logger.info(f"Section: {args.section}")
    if args.letter:
        logger.info(f"Filter letter: {args.letter}")

    async with BrowserManager(headless=not args.show_browser, delay=args.delay) as browser:
        count = await scrape_section(
            section=args.section,
            browser=browser,
            data_dir=data_dir,
            progress_dir=progress_dir,
            letter=args.letter,
        )

    logger.info("=" * 50)
    logger.info(f"Scraping complete! Total entries: {count}")
    logger.info("=" * 50)

    return count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape UNAM Traditional Mexican Medicine Digital Library"
    )
    parser.add_argument(
        "--section",
        choices=["dictionary", "plants", "peoples", "flora", "all"],
        default="all",
        help="Section to scrape (default: all)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--letter",
        "-l",
        help="Only scrape entries starting with this letter (for testing)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=1.5,
        help="Delay between requests in seconds (default: 1.5)",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show browser window (for debugging)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Scraping failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
