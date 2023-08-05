import pandas as pd

from .util import COLLECTROOT

AGES_DATE_FMT = "%d.%m.%Y"
HMS_TIME = "%H:%M:%S"
AGES_TIME_FMT = f"{AGES_DATE_FMT} {HMS_TIME}"
ISO_DATE_FMT = "%Y-%m-%d"
ISO_SP_TIME_FMT = f"{ISO_DATE_FMT} {HMS_TIME}"
ISO_TIME_FMT = f"{ISO_DATE_FMT}T{HMS_TIME}"
ISO_TIME_TZ_FMT = f"{ISO_TIME_FMT}%z"


def load_ww_blverlauf() -> pd.DataFrame:
    blverlauf = pd.read_csv(
        COLLECTROOT / "covid/abwassermonitoring/blverlauf_all.csv.xz", sep=";", parse_dates=["FileDate", "Datum"]
    )
    blverlauf.rename(columns={"name": "Bundesland"}, inplace=True, errors="raise")
    return blverlauf


def first_filedate(df: pd.DataFrame, catcols: list[str]) -> pd.DataFrame:
    return df.sort_values("FileDate", kind="stable").groupby(["Datum", *catcols]).first()


def add_date(df: pd.DataFrame, colname: str, format=None) -> pd.DataFrame:  # noqa: A002
    df["Datum"] = pd.to_datetime(df[colname], dayfirst=True, format=format, exact=format is not None)
    if colname != "Datum":
        df.drop(columns=colname, inplace=True)
    return df


def norm_df(df: pd.DataFrame, *, datecol: str, format=None) -> pd.DataFrame:  # noqa: A002
    add_date(df, datecol, format=format)
    return df


def shorten_bezname(bezname: str, *, soft=False) -> str:
    result = (
        bezname.replace("Sankt Johann im Pongau", "Pongau")
        .replace(" an der ", "/")
        .replace(" am ", "/")
        .replace(" im ", "/")
        .replace("Sankt ", "St. ")
        .replace(" Stadt", "")
        .replace("Wiener ", "Wr. ")
    )
    if not soft or not result.startswith("Salzburg"):
        result = result.replace("(Stadt)", "")
    if not soft:
        result = (
            result.replace("-Stadt", "")
            .replace("stadt", "st.")
            .replace("dorf", "df.")
            .replace("-Land", "-L")  # Landeck must not become Leck
            .replace(" Land", " L")
            .replace("(Land)", " L")
            .replace("Umgebung", "U")
            .replace("Südoststeiermark", "SO-Stmk.")
            .replace("Bruck-Mürzzuschlag", "Bruck-Mzz.")
            .replace("Hartberg-Fürstenfeld", "Hartb.-Ff.")
            .replace("Innsbruck", "Ibk.")
            .replace("Deutschlandsberg", "DE-berg")
            .replace("Waidhofen", "Waidh.")
        )
    return result.strip()


SHORTNAME_BY_BUNDESLAND = {
    "Burgenland": "B",
    "Kärnten": "K",
    "Niederösterreich": "N",
    "Oberösterreich": "O",
    "Salzburg": "Sa",
    "Steiermark": "St",
    "Tirol": "T",
    "Vorarlberg": "V",
    "Wien": "W",
    "Österreich": "A",
}

SHORTNAME2_BY_BUNDESLAND = {
    "Burgenland": "Bgld.",
    "Kärnten": "Ktn.",
    "Niederösterreich": "NÖ",
    "Oberösterreich": "OÖ",
    "Salzburg": "Sbg.",
    "Steiermark": "Stmk.",
    "Tirol": "Tirol",
    "Vorarlberg": "Vbg.",
    "Wien": "Wien",
    "Österreich": "AUT",
}
