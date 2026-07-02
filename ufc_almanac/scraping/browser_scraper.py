import bs4
import time
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from ufc_almanac.globals import BLOCKED_RESOURCE_TYPES


class BrowserScraper:
    """
    Fetches pages through a headless Chromium browser so JavaScript challenges
    (e.g. ufcstats.com bot checks) can run before HTML is parsed.
    """

    def __init__(
        self,
        headless=True,
        timeout_ms=15_000,
        max_retries=2,
        retry_delay_s=2.0,
    ):
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self._playwright = None
        self._browser = None
        self._page = None

    def __enter__(self):
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page()
        self._page.route(
            "**/*",
            lambda route: (
                route.abort()
                if route.request.resource_type in BLOCKED_RESOURCE_TYPES
                else route.continue_()
            ),
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def get_soup(
        self,
        url,
        wait_selector=None,
        timeout_ms=None,
        retries=None,
        selector_state="attached",
    ):
        """
        Load a URL in the browser and return a BeautifulSoup object for the
        rendered HTML. Optionally wait for a CSS selector before parsing.
        """
        timeout = self.timeout_ms if timeout_ms is None else timeout_ms
        attempts = (self.max_retries if retries is None else retries) + 1
        last_error = None

        for attempt in range(attempts):
            try:
                self._page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                if wait_selector:
                    self._page.wait_for_selector(
                        wait_selector,
                        timeout=timeout,
                        state=selector_state,
                    )
                return bs4.BeautifulSoup(self._page.content(), "lxml")
            except PlaywrightTimeoutError as error:
                last_error = error
                if attempt < attempts - 1:
                    time.sleep(self.retry_delay_s)
                    continue
                raise last_error from None
