import datetime
import os
import pandas

from ufc_almanac.globals import RESULTS_CSV, STATS_CSV, VERBOSE


# Local #

def _parse_csv_date(date_text: str) -> datetime.date:
    """Parse a stored fight date such as '20/11/2021'."""
    return datetime.datetime.strptime(str(date_text).strip(), "%d/%m/%Y").date()

def _parse_site_event_date(date_text: str) -> datetime.date:
    """Parse an event listing date such as 'April 07, 1995'."""
    return datetime.datetime.strptime(date_text.strip(), "%B %d, %Y").date()

def _reorder_fight_stats_to_results(results_dataframe, stats_dataframe):
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

def _sort_fight_results_chronologically(results_dataframe):
    """Return fight results with the oldest fights first."""
    sorted_results = results_dataframe.copy()
    sorted_results["_sort_date"] = sorted_results["Date"].map(_parse_csv_date)
    sorted_results = sorted_results.sort_values("_sort_date", kind="stable")
    return sorted_results.drop(columns="_sort_date").reset_index(drop=True)

def _strip_fighter_name_columns(results_dataframe, stats_dataframe):
    """Strip whitespace from fighter name columns in both dataframes."""
    cleaned_results = results_dataframe.copy()
    cleaned_stats = stats_dataframe.copy()

    for column in ("Fighter 1", "Fighter 2"):
        if column in cleaned_results.columns:
            cleaned_results[column] = cleaned_results[column].map(clean_fighter_name)

    if "Name" in cleaned_stats.columns:
        cleaned_stats["Name"] = cleaned_stats["Name"].map(clean_fighter_name)

    return cleaned_results, cleaned_stats


# Global #

def clean_fighter_name(name_text):
    """Remove leading/trailing whitespace and collapse internal spacing."""
    return " ".join(str(name_text).split())

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

def filter_new_events(events, latest_date):
    """
    Keep only events that occur after the latest date already in the CSV files.
    """
    if latest_date is None:
        return events
    return [(url, event_date) for url, event_date in events if event_date > latest_date]

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
        latest_dates.append(max(_parse_csv_date(value) for value in dataframe["Date"]))
    return max(latest_dates) if latest_dates else None

def load_fight_csv(path):
    """Load a fight CSV file, dropping any legacy index column."""
    if not os.path.exists(path):
        return None

    dataframe = pandas.read_csv(path)
    if "Unnamed: 0" in dataframe.columns:
        dataframe = dataframe.drop(columns=["Unnamed: 0"])
    return dataframe

def normalize_fight_dataframes(results_dataframe, stats_dataframe):
    """Sort results oldest-first and reorder stats to match."""
    cleaned_results, cleaned_stats = _strip_fighter_name_columns(
        results_dataframe,
        stats_dataframe,
    )
    sorted_results = _sort_fight_results_chronologically(cleaned_results)
    sorted_stats = _reorder_fight_stats_to_results(sorted_results, cleaned_stats)
    return sorted_results, sorted_stats

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
    for parser in (_parse_csv_date, _parse_site_event_date):
        try:
            return parser(date_text)
        except ValueError:
            continue

    raise ValueError(
        f"Could not parse date {date_value!r}; use dd/mm/yyyy or 'Month DD, YYYY'"
    )

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
            event_date = _parse_site_event_date(date_element.get_text(strip=True))
        except ValueError:
            continue

        if url not in seen_urls:
            seen_urls.add(url)
            events.append((url, event_date))

    return events

def parse_next_event(soup) -> tuple[str, datetime.date]:
    """
    Find the URL and date of the next upcoming UFC event.
    """
    for link in soup.find_all(
        "a", href=True, attrs={"class": "b-link b-link_style_white"}
    ):
        try:
            url = link["href"]
            if "event-details" not in url: continue

            parent = link.find_parent("i")
            if parent is None: continue

            date_element = parent.find("span", attrs={"class": "b-statistics__date"})
            if date_element is None: continue

            event_date = _parse_site_event_date(date_element.get_text(strip=True))
            return url, event_date
        except Exception as e:
            if VERBOSE: print(f"Error parsing next event: {e}")
            continue

    raise ValueError("No next event found")
