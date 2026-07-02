import datetime
import pandas
import string
from tqdm import tqdm

from ufc_almanac.globals import FIGHTER_DATA_CSV
from ufc_almanac.scraping.browser_scraper import BrowserScraper
from ufc_almanac.scraping.utils import clean_fighter_name


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


def main() -> None:
    scrape_fighter_data()


if __name__ == "__main__":
    main()
