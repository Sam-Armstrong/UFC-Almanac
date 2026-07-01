import bs4
import datetime
import os
import pandas
import string
import time
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from tqdm import tqdm

from globals import FIGHTER_DATA_CSV, RESULTS_CSV, STATS_CSV, VERBOSE


def parse_site_event_date(date_text):
    """Parse an event listing date such as 'April 07, 1995'."""
    return datetime.datetime.strptime(date_text.strip(), "%B %d, %Y").date()


def parse_csv_date(date_text):
    """Parse a stored fight date such as '20/11/2021'."""
    return datetime.datetime.strptime(str(date_text).strip(), "%d/%m/%Y").date()


def clean_fighter_name(name_text):
    """Remove leading/trailing whitespace and collapse internal spacing."""
    return " ".join(str(name_text).split())


def strip_fighter_name_columns(results_dataframe, stats_dataframe):
    """Strip whitespace from fighter name columns in both dataframes."""
    cleaned_results = results_dataframe.copy()
    cleaned_stats = stats_dataframe.copy()

    for column in ("Fighter 1", "Fighter 2"):
        if column in cleaned_results.columns:
            cleaned_results[column] = cleaned_results[column].map(clean_fighter_name)

    if "Name" in cleaned_stats.columns:
        cleaned_stats["Name"] = cleaned_stats["Name"].map(clean_fighter_name)

    return cleaned_results, cleaned_stats


def get_latest_scraped_date(results_path=RESULTS_CSV, stats_path=STATS_CSV):
    """
    Return the most recent fight date already stored in the CSV files, or None
    if no data exists yet.
    """
    latest_dates = list()
    for path in (results_path, stats_path):
        if not os.path.exists(path):
            continue
        dataframe = pandas.read_csv(path)
        if dataframe.empty:
            continue
        latest_dates.append(max(parse_csv_date(value) for value in dataframe["Date"]))
    return max(latest_dates) if latest_dates else None


def parse_event_listing(soup):
    """
    Extract event URLs and dates from the completed-events listing page.
    """
    events = list()
    seen_urls = set()

    for link in soup.find_all(
        "a", href=True, attrs={"class": "b-link b-link_style_black"}
    ):
        url = link["href"]
        if "event-details" not in url:
            continue

        parent = link.find_parent("i")
        if parent is None:
            continue

        date_element = parent.find("span", attrs={"class": "b-statistics__date"})
        if date_element is None:
            continue

        try:
            event_date = parse_site_event_date(date_element.get_text(strip=True))
        except ValueError:
            continue

        if url not in seen_urls:
            seen_urls.add(url)
            events.append((url, event_date))

    return events


def filter_new_events(events, latest_date):
    """
    Keep only events that occur after the latest date already in the CSV files.
    """
    if latest_date is None:
        return events
    return [(url, event_date) for url, event_date in events if event_date > latest_date]


def parse_cutoff_date(date_value):
    """
    Parse an optional cutoff date from a date object or string.

    Strings may use dd/mm/yyyy (CSV format) or 'Month DD, YYYY' (site format).
    """
    if date_value is None:
        return None
    if isinstance(date_value, datetime.date):
        return date_value

    date_text = str(date_value).strip()
    for parser in (parse_csv_date, parse_site_event_date):
        try:
            return parser(date_text)
        except ValueError:
            continue

    raise ValueError(
        f"Could not parse date {date_value!r}; use dd/mm/yyyy or 'Month DD, YYYY'"
    )


def filter_events_by_date_range(events, start_date=None, end_date=None):
    """Keep only events within the optional inclusive start/end date range."""
    filtered = events
    if start_date is not None:
        filtered = [
            (url, event_date)
            for url, event_date in filtered
            if event_date >= start_date
        ]
    if end_date is not None:
        filtered = [
            (url, event_date) for url, event_date in filtered if event_date <= end_date
        ]
    return filtered


def sort_fight_results_chronologically(results_dataframe):
    """Return fight results with the oldest fights first."""
    sorted_results = results_dataframe.copy()
    sorted_results["_sort_date"] = sorted_results["Date"].map(parse_csv_date)
    sorted_results = sorted_results.sort_values("_sort_date", kind="stable")
    return sorted_results.drop(columns="_sort_date").reset_index(drop=True)


def reorder_fight_stats_to_results(results_dataframe, stats_dataframe):
    """
    Align fight stats row order with fight results, keeping each fighter pair
    together for a given fight.
    """
    stats_remaining = stats_dataframe.copy().reset_index(drop=True)
    if "Unnamed: 0" in stats_remaining.columns:
        stats_remaining = stats_remaining.drop(columns=["Unnamed: 0"])

    stats_remaining["Name"] = stats_remaining["Name"].astype(str).str.strip()
    stats_remaining["_date"] = stats_remaining["Date"].astype(str).str.strip()
    ordered_rows = list()

    for _, result in results_dataframe.iterrows():
        fight_date = str(result["Date"]).strip()
        fighter1 = str(result["Fighter 1"]).strip()
        fighter2 = str(result["Fighter 2"]).strip()

        fighter1_match = stats_remaining[
            (stats_remaining["_date"] == fight_date)
            & (stats_remaining["Name"] == fighter1)
        ]
        fighter2_match = stats_remaining[
            (stats_remaining["_date"] == fight_date)
            & (stats_remaining["Name"] == fighter2)
        ]

        if fighter1_match.empty or fighter2_match.empty:
            continue

        fighter1_index = fighter1_match.index[0]
        fighter2_index = fighter2_match.index[0]
        ordered_rows.append(stats_remaining.loc[fighter1_index].drop(labels=["_date"]))
        ordered_rows.append(stats_remaining.loc[fighter2_index].drop(labels=["_date"]))
        stats_remaining = stats_remaining.drop([fighter1_index, fighter2_index])

    if not stats_remaining.empty:
        for _, row in stats_remaining.iterrows():
            ordered_rows.append(row.drop(labels=["_date"]))

    return pandas.DataFrame(ordered_rows).reset_index(drop=True)


def normalize_fight_dataframes(results_dataframe, stats_dataframe):
    """Sort results oldest-first and reorder stats to match."""
    cleaned_results, cleaned_stats = strip_fighter_name_columns(
        results_dataframe,
        stats_dataframe,
    )
    sorted_results = sort_fight_results_chronologically(cleaned_results)
    sorted_stats = reorder_fight_stats_to_results(sorted_results, cleaned_stats)
    return sorted_results, sorted_stats


def load_fight_csv(path):
    """Load a fight CSV file, dropping any legacy index column."""
    if not os.path.exists(path):
        return None

    dataframe = pandas.read_csv(path)
    if "Unnamed: 0" in dataframe.columns:
        dataframe = dataframe.drop(columns=["Unnamed: 0"])
    return dataframe


BLOCKED_RESOURCE_TYPES = {"image", "media", "font"}


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


def scrape_past_fights(start_date=None, end_date=None):
    """
    Scrapes all the data for past fights from the internet and stores this in
    separate results and stats CSV files.

    Optional start_date and end_date limit which events are scraped (inclusive).
    Accept date objects or strings in dd/mm/yyyy or 'Month DD, YYYY' format.
    """

    start_cutoff = parse_cutoff_date(start_date)
    end_cutoff = parse_cutoff_date(end_date)
    if (
        start_cutoff is not None
        and end_cutoff is not None
        and start_cutoff > end_cutoff
    ):
        raise ValueError("start_date must be on or before end_date")

    def run(scraper):
        initial_url = "http://www.ufcstats.com/statistics/events/completed?page=all"
        soup = scraper.get_soup(
            initial_url, wait_selector="a.b-link.b-link_style_black"
        )
        url_list = list()

        latest_date = get_latest_scraped_date()
        all_events = parse_event_listing(soup)
        new_events = filter_new_events(all_events, latest_date)
        events_before_range_filter = len(new_events)
        new_events = filter_events_by_date_range(new_events, start_cutoff, end_cutoff)
        new_events.sort(key=lambda event: event[1])
        event_urls = [url for url, _ in new_events]

        range_parts = list()
        if start_cutoff is not None:
            range_parts.append(f"from {start_cutoff.strftime('%d/%m/%Y')}")
        if end_cutoff is not None:
            range_parts.append(f"through {end_cutoff.strftime('%d/%m/%Y')}")
        range_label = " ".join(range_parts)

        if latest_date is None:
            print(f"Found {len(event_urls)} events to scrape.")
        else:
            print(
                f"Latest scraped date: {latest_date.strftime('%d/%m/%Y')} "
                f"({len(all_events) - events_before_range_filter} events skipped, "
                f"{len(event_urls)} new events to scrape)"
            )

        if range_label:
            skipped_by_range = events_before_range_filter - len(new_events)
            print(
                f"Date range filter ({range_label}): "
                f"{skipped_by_range} events outside range skipped, "
                f"{len(event_urls)} events to scrape"
            )

        if not event_urls:
            print("No new events to scrape.")
            return

        for url in tqdm(event_urls, desc="Collecting fights from events", unit="event"):
            soup = scraper.get_soup(
                url,
                wait_selector="tr.b-fight-details__table-row",
            )
            for link in soup.find_all(
                "tr",
                attrs={
                    "class": "b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click"
                },
            ):
                url_list.append(link["data-link"])

        results_dataframe = pandas.DataFrame(
            columns=["Date", "Fighter 1", "Fighter 2", "Result", "Split Dec?"]
        )
        stats_dataframe = pandas.DataFrame(
            columns=[
                "Name",
                "Date",
                "Time",
                "Knockdowns",
                "Knockdowns Against",
                "Sig Strikes Landed",
                "Sig Strikes Attempted",
                "Sig Strikes Absorbed",
                "Strikes Landed",
                "Strikes Attempted",
                "Strikes Absorbed",
                "Takedowns",
                "Takedown Attempts",
                "Got Taken Down",
                "Submission Attempts",
                "Clinch Strikes",
                "Clinch Strikes Taken",
                "Ground Strikes",
                "Ground Strikes Taken",
            ]
        )

        all_info = list()
        all_stats = list()

        for url in tqdm(url_list, desc="Scraping fights", unit="fight"):
            try:
                soup = scraper.get_soup(
                    url,
                    wait_selector="div.b-fight-details",
                    timeout_ms=30_000,
                    retries=1,
                )
                if not soup.find("p", class_="b-fight-details__table-text"):
                    tqdm.write(f"Skipping (no detailed stats): {url}")
                    continue

                stats = list()
                fighter1_stats = list()
                fighter2_stats = list()
                info_data = list()
                split_dec = 0

                text1 = soup.find_all(
                    "i", attrs={"class": "b-fight-details__text-item_first"}
                )
                text2 = soup.find_all(
                    "i",
                    attrs={
                        "class": "b-fight-details__person-status b-fight-details__person-status_style_gray"
                    },
                )

                if "Decision - S" in text1[0].text or len(text2) > 1:
                    split_dec = 1

                raw_stats = soup.find_all(
                    "p", attrs={"class": "b-fight-details__table-text"}
                )
                for element in raw_stats:
                    stat = element.text
                    stat = stat.replace("\n", "")
                    stat = stat.replace("  ", "")
                    stats.append(stat)

                name1 = clean_fighter_name(stats[0])
                name2 = clean_fighter_name(stats[1])

                new_fight_info = list()
                fight_info = soup.find_all(
                    "i", attrs={"class": "b-fight-details__text-item"}
                )
                for info in fight_info:
                    info = info.text
                    info = info.replace("\n", "")
                    info = info.replace(" ", "")
                    new_fight_info.append(info)

                rounds = new_fight_info[0].split(":")[1]

                time_list = new_fight_info[1].split(":")
                time_min = time_list[1]
                time_sec = time_list[2]

                time = ((int(rounds) - 1) * 300) + (int(time_min) * 60) + int(time_sec)

                knockdowns1 = stats[2]
                knockdowns2 = stats[3]

                sig_strike1_list = stats[4].split(" of ")
                sig_strikes_landed1 = sig_strike1_list[0]
                sig_strikes_attempted1 = sig_strike1_list[1]
                sig_strike2_list = stats[5].split(" of ")
                sig_strikes_landed2 = sig_strike2_list[0]
                sig_strikes_attempted2 = sig_strike2_list[1]

                strike1_list = stats[8].split(" of ")
                strikes_landed1 = strike1_list[0]
                strikes_attempted1 = strike1_list[1]
                strike2_list = stats[9].split(" of ")
                strikes_landed2 = strike2_list[0]
                strikes_attempted2 = strike2_list[1]

                takedowns1 = stats[10].split(" of ")[0]
                takedown_attempts1 = stats[10].split(" of ")[1]
                takedowns2 = stats[11].split(" of ")[0]
                takedown_attempts2 = stats[11].split(" of ")[1]

                submission_attempts1 = stats[14]
                submission_attempts2 = stats[15]

                clinch_strikes1 = stats[34 + (int(rounds) * 20)].split(" of ")[0]
                clinch_strikes2 = stats[35 + (int(rounds) * 20)].split(" of ")[0]

                ground_strikes1 = stats[36 + (int(rounds) * 20)].split(" of ")[0]
                ground_strikes2 = stats[37 + (int(rounds) * 20)].split(" of ")[0]

                win_loss = soup.find_all(
                    "div", attrs={"class": "b-fight-details__person"}
                )
                for w in win_loss:
                    l = w.text.replace("\n", "")
                    l = l.split(" ")

                    for x in l:
                        if x == "W":
                            r = 2
                        if x == "L":
                            r = 1
                        if x == "D":
                            r = 3
                        if x == "NC":
                            r = 4
                            if VERBOSE: tqdm.write(f"No Contest: {name1} vs {name2}")

                date_element = soup.find_all("a", href=True, attrs={"class": "b-link"})[
                    0
                ]
                date_url = date_element["href"]
                soup = scraper.get_soup(
                    date_url,
                    wait_selector="li.b-list__box-list-item",
                    timeout_ms=30_000,
                    retries=1,
                )
                raw_date = soup.find_all(
                    "li", attrs={"class": "b-list__box-list-item"}
                )[0].text
                raw_date = raw_date.replace("\n", "")
                raw_date = raw_date.replace("Date:", "")
                raw_date = raw_date.replace("  ", "")
                raw_date = raw_date.replace(",", "")
                date_list = raw_date.split(" ")
                month = datetime.datetime.strptime(date_list[0], "%B").month
                day = date_list[1]
                year = date_list[2]

                date = str(day) + "/" + str(month) + "/" + str(year)

                info_data.append(date)
                info_data.append(name1)
                info_data.append(name2)
                info_data.append(r)
                info_data.append(split_dec)

                fighter1_stats.append(name1)
                fighter1_stats.append(date)
                fighter1_stats.append(time)
                fighter1_stats.append(knockdowns1)
                fighter1_stats.append(knockdowns2)
                fighter1_stats.append(sig_strikes_landed1)
                fighter1_stats.append(sig_strikes_attempted1)
                fighter1_stats.append(sig_strikes_landed2)
                fighter1_stats.append(strikes_landed1)
                fighter1_stats.append(strikes_attempted1)
                fighter1_stats.append(strikes_landed2)
                fighter1_stats.append(takedowns1)
                fighter1_stats.append(takedown_attempts1)
                fighter1_stats.append(takedowns2)
                fighter1_stats.append(submission_attempts1)
                fighter1_stats.append(clinch_strikes1)
                fighter1_stats.append(clinch_strikes2)
                fighter1_stats.append(ground_strikes1)
                fighter1_stats.append(ground_strikes2)

                fighter2_stats.append(name2)
                fighter2_stats.append(date)
                fighter2_stats.append(time)
                fighter2_stats.append(knockdowns2)
                fighter2_stats.append(knockdowns1)
                fighter2_stats.append(sig_strikes_landed2)
                fighter2_stats.append(sig_strikes_attempted2)
                fighter2_stats.append(sig_strikes_landed1)
                fighter2_stats.append(strikes_landed2)
                fighter2_stats.append(strikes_attempted2)
                fighter2_stats.append(strikes_landed1)
                fighter2_stats.append(takedowns2)
                fighter2_stats.append(takedown_attempts2)
                fighter2_stats.append(takedowns1)
                fighter2_stats.append(submission_attempts2)
                fighter2_stats.append(clinch_strikes2)
                fighter2_stats.append(clinch_strikes1)
                fighter2_stats.append(ground_strikes2)
                fighter2_stats.append(ground_strikes1)

                all_info.append(info_data)
                all_stats.append(fighter1_stats)
                all_stats.append(fighter2_stats)

            except PlaywrightTimeoutError:
                tqdm.write(f"Skipping (page timeout): {url}")
            except Exception as e:
                tqdm.write(f"Passing: {url} ({e})")

        for data in all_info:
            df_len = len(results_dataframe)
            results_dataframe.loc[df_len] = data

        for stat in all_stats:
            df_len = len(stats_dataframe)
            stats_dataframe.loc[df_len] = stat

        if results_dataframe.empty and stats_dataframe.empty:
            print("No new fight data collected.")
            return

        existing_results = load_fight_csv(RESULTS_CSV)
        if existing_results is not None:
            results_dataframe = pandas.concat(
                [existing_results, results_dataframe],
                ignore_index=True,
            )

        existing_stats = load_fight_csv(STATS_CSV)
        if existing_stats is not None:
            stats_dataframe = pandas.concat(
                [existing_stats, stats_dataframe],
                ignore_index=True,
            )

        results_dataframe, stats_dataframe = normalize_fight_dataframes(
            results_dataframe,
            stats_dataframe,
        )

        results_dataframe.to_csv(RESULTS_CSV, index=False)
        stats_dataframe.to_csv(STATS_CSV, index=False)
        print(f"Saved {len(all_info)} new fights to {RESULTS_CSV} and {STATS_CSV}")

    with BrowserScraper() as scraper:
        run(scraper)


def scrape_fighter_data():
    """
    Scrapes the data for each individual fighter from the internet and stores
    this in a CSV file.
    """

    def calculate_age(month, day, year):
        today = datetime.date.today()
        return today.year - year - ((today.month, today.day) < (month, day))

    def run(scraper):
        not_allowed = [
            "%",
            "lbs.",
            ",",
            '"',
            "Record:",
            "Reach:",
            "SLpM:",
            "Str. Acc.:",
            "SApM:",
            "Str. Def:",
            "TD Avg.:",
            "TD Acc.:",
            "TD Def.:",
            "Sub. Avg.:",
            " \n",
            "\n",
            "  ",
        ]
        url_list = list()

        for c in tqdm(
            string.ascii_lowercase, desc="Collecting fighter URLs", unit="letter"
        ):
            url = "http://www.ufcstats.com/statistics/fighters?char=%s&page=all" % c
            soup = scraper.get_soup(url, wait_selector="a.b-link.b-link_style_black")
            for a in soup.find_all(
                "a", href=True, attrs={"class": "b-link b-link_style_black"}
            ):
                new_url = a["href"]
                if new_url not in url_list:
                    url_list.append(new_url)

        fighters_dataframe = pandas.DataFrame(
            columns=[
                "Name",
                "Wins",
                "Losses",
                "Draws",
                "Height",
                "Weight",
                "Reach",
                "Stance",
                "Age",
                "SLpM",
                "StrAcc",
                "SApM",
                "StrDef",
                "TDAvg",
                "TDAcc",
                "TDDef",
                "SubAvg",
            ]
        )

        for url in tqdm(url_list, desc="Scraping fighters", unit="fighter"):
            soup = scraper.get_soup(url, wait_selector="h2")

            data = list()
            for h2 in soup.find_all("h2"):
                raw_fighter_name = h2.find(
                    "span", attrs={"class": "b-content__title-highlight"}
                ).text

                for i in range(0, len(not_allowed)):
                    if not_allowed[i] in raw_fighter_name:
                        raw_fighter_name = raw_fighter_name.replace(not_allowed[i], "")

                name = clean_fighter_name(raw_fighter_name)

                record = h2.find(
                    "span", attrs={"class": "b-content__title-record"}
                ).text
                for i in [" ", "\n", "Record:"]:
                    if i in record:
                        record = record.replace(i, "")

                record = record.split("-")

                wins = record[0]
                losses = record[1]
                draws = record[2]

                if "(" in draws:
                    draws = draws[0]

                data.append(name)
                data.append(wins)
                data.append(losses)
                data.append(draws)

            for ul in soup.find_all("ul"):
                for li in ul.find_all(
                    "li",
                    attrs={
                        "class": "b-list__box-list-item b-list__box-list-item_type_block"
                    },
                ):
                    collected_data = li.text

                    for i in range(0, len(not_allowed)):
                        if not_allowed[i] in collected_data:
                            collected_data = collected_data.replace(not_allowed[i], "")

                    if ("Height:" in str(collected_data)) and (
                        "--" not in str(collected_data)
                    ):
                        collected_data = collected_data.replace("Height:", "")
                        measurement = collected_data.split("'")

                        cm1 = int(measurement[0]) * 30.48
                        cm2 = int(measurement[1]) * 2.54
                        collected_data = round((cm1 + cm2), 1)

                    if ("DOB:" in str(collected_data)) and (
                        "--" not in str(collected_data)
                    ):
                        collected_data = collected_data.replace("DOB:", "")
                        dateList = collected_data.split(" ")
                        monthStr = str(dateList[0])
                        day = int(dateList[1])
                        year = int(dateList[2])
                        month = datetime.datetime.strptime(monthStr, "%b").month
                        collected_data = int(calculate_age(month, day, year))

                    if "Weight:" in str(collected_data):
                        collected_data = collected_data.replace("Weight:", "")
                        collected_data = collected_data.replace("lbs", "")
                        collected_data = collected_data.replace(" ", "")

                    elif "STAN" in str(collected_data):
                        if "Orthodox" in collected_data:
                            collected_data = 1
                        elif "Southpaw" in collected_data:
                            collected_data = 2
                        else:
                            collected_data = 3

                    collected_data = str(collected_data)

                    if (collected_data != "") and ("--" not in collected_data):
                        data.append(collected_data)

            if len(data) == 17:
                df_len = len(fighters_dataframe)
                fighters_dataframe.loc[df_len] = data

        if "Name" in fighters_dataframe.columns:
            fighters_dataframe["Name"] = fighters_dataframe["Name"].map(
                clean_fighter_name
            )

        fighters_dataframe.to_csv(FIGHTER_DATA_CSV)

    with BrowserScraper() as scraper:
        run(scraper)


if __name__ == "__main__":
    scrape_past_fights(start_date="1/1/2010")
    scrape_fighter_data()
