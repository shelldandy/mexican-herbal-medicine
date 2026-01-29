"""Playwright browser management for scraping."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright, Browser, Page, BrowserContext

logger = logging.getLogger(__name__)


class BrowserManager:
    """Manages Playwright browser instance for scraping."""

    def __init__(self, headless: bool = True, delay: float = 1.5):
        """
        Initialize browser manager.

        Args:
            headless: Run browser in headless mode
            delay: Delay between requests in seconds
        """
        self.headless = headless
        self.delay = delay
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def __aenter__(self):
        """Start browser on context enter."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close browser on context exit."""
        await self.close()

    async def start(self) -> None:
        """Start the browser."""
        logger.info("Starting browser...")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
        )
        self.page = await self.context.new_page()
        logger.info("Browser started successfully")

    async def close(self) -> None:
        """Close the browser."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("Browser closed")

    async def get_page_content(self, url: str, wait_selector: str = None) -> str:
        """
        Navigate to URL and return page content.

        Args:
            url: URL to navigate to
            wait_selector: Optional CSS selector to wait for before returning content

        Returns:
            HTML content of the page
        """
        logger.debug(f"Fetching: {url}")

        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)

            if wait_selector:
                try:
                    await self.page.wait_for_selector(wait_selector, timeout=10000)
                except Exception:
                    logger.warning(f"Selector {wait_selector} not found on {url}")

            # Small delay to ensure dynamic content loads
            await asyncio.sleep(0.5)

            content = await self.page.content()

            # Respectful delay between requests
            await asyncio.sleep(self.delay)

            return content

        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            raise

    async def get_links(self, url: str, selector: str) -> list[dict]:
        """
        Get all links matching a selector from a page.

        Args:
            url: URL to navigate to
            selector: CSS selector for links

        Returns:
            List of dicts with 'href' and 'text' keys
        """
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(0.5)

        links = await self.page.evaluate(f"""
            () => {{
                const links = document.querySelectorAll('{selector}');
                return Array.from(links).map(a => ({{
                    href: a.href,
                    text: a.textContent.trim()
                }}));
            }}
        """)

        await asyncio.sleep(self.delay)
        return links

    async def download_image(self, url: str, save_path: Path) -> bool:
        """
        Download an image to the specified path.

        Args:
            url: URL of the image
            save_path: Path to save the image

        Returns:
            True if successful, False otherwise
        """
        import aiohttp
        import aiofiles

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        async with aiofiles.open(save_path, 'wb') as f:
                            await f.write(await response.read())
                        logger.debug(f"Downloaded image: {save_path}")
                        return True
                    else:
                        logger.warning(f"Failed to download {url}: HTTP {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False
