import datetime
import pandas
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from tqdm import tqdm

from ufc_almanac.globals import RESULTS_CSV, STATS_CSV, VERBOSE
from ufc_almanac.scraping.browser_scraper import BrowserScraper
from ufc_almanac.scraping.utils import (
    clean_fighter_name,
    get_latest_scraped_date,
    parse_cutoff_date,
    filter_events_by_date_range,
    filter_new_events,
    parse_event_listing,
    normalize_fight_dataframes,
    load_fight_csv,
)


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
            columns=[
                "Date",
                "Fighter 1",
                "Fighter 2",
                "Result",
                "Method",
            ]
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

                method_text = soup.find_all(
                    "i", attrs={"class": "b-fight-details__text-item_first"}
                )

                method = str(method_text[0].text).replace("Method:", "").strip()

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
                info_data.append(method)

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


def main() -> None:
    scrape_past_fights(start_date="1/1/2010")


if __name__ == "__main__":
    main()
