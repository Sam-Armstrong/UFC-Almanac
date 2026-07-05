import datetime
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from tqdm import tqdm

from ufc_almanac.scraping.browser_scraper import BrowserScraper
from ufc_almanac.scraping.utils import parse_next_event


class UpcomingEvent:
    def __init__(self, url: str, date: datetime.date, fights: list[tuple[str, str]]):
        self.url = url
        self.date = date
        self.fights: list[tuple[str, str]] = fights

    def __str__(self) -> str:
        return f"Event(url={self.url}, date={self.date}, fights={self.fights})"

    def __repr__(self) -> str:
        return self.__str__()


def scrape_next_event() -> UpcomingEvent:
    """
    Scrape the fighters and date of the next upcoming UFC event.
    """

    def run(scraper):
        initial_url = "http://www.ufcstats.com/statistics/events/completed?page=all"
        soup = scraper.get_soup(
            initial_url, wait_selector="a.b-link.b-link_style_black"
        )

        next_event_url, next_event_date = parse_next_event(soup)
        print(f"Next event: {next_event_url} on {next_event_date}")

        fights: list[tuple[str, str]] = []
        current_fight: list[str] = []

        try:
            soup = scraper.get_soup(
                next_event_url,
                wait_selector="div.b-fight-details",
                timeout_ms=30_000,
                retries=1,
            )

            for fighter_link in soup.find_all(
                "a", href=True, attrs={"class": "b-link b-link_style_black"}
            ):
                fighter_name = fighter_link.get_text(strip=True)
                current_fight.append(fighter_name)
                if len(current_fight) == 2:
                    fights.append(tuple(current_fight))
                    current_fight = []

            return UpcomingEvent(next_event_url, next_event_date, fights)
        except PlaywrightTimeoutError:
            tqdm.write(f"Skipping (page timeout): {next_event_url}")
        except Exception as e:
            tqdm.write(f"Passing: {next_event_url} ({e})")

    with BrowserScraper() as scraper:
        return run(scraper)


def main() -> None:
    event = scrape_next_event()
    print(event)


if __name__ == "__main__":
    main()
