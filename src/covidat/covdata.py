from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from .util import COLLECTROOT, DATAROOT, fdate_from_fname

AGES_DATE_FMT = "%d.%m.%Y"
HMS_TIME = "%H:%M:%S"
AGES_TIME_FMT = f"{AGES_DATE_FMT} {HMS_TIME}"
ISO_DATE_FMT = "%Y-%m-%d"
ISO_SP_TIME_FMT = f"{ISO_DATE_FMT} {HMS_TIME}"
ISO_TIME_FMT = f"{ISO_DATE_FMT}T{HMS_TIME}"
ISO_TIME_TZ_FMT = f"{ISO_TIME_FMT}%z"


def loadall_csv(fname_glob: str, *, min_distance: timedelta = timedelta(0)) -> pd.DataFrame:
    frames = []
    prev_fdate: datetime | None = None
    for p in sorted(DATAROOT.glob(fname_glob), reverse=True):
        fdate = fdate_from_fname(p)
        if prev_fdate is None or prev_fdate - fdate >= min_distance:
            frames.append(pd.read_csv(p, sep=";").assign(FileDate=fdate))
        prev_fdate = fdate
    return pd.concat(
        frames[::-1],
    )


def load_ww_blverlauf() -> pd.DataFrame:
    blverlauf = pd.read_csv(
        COLLECTROOT / "covid/abwassermonitoring/blverlauf_all.csv.xz", sep=";", parse_dates=["FileDate", "Datum"]
    )
    blverlauf.rename(columns={"name": "Bundesland"}, inplace=True, errors="raise")
    return blverlauf


def first_filedate(df: pd.DataFrame, catcols: list[str]) -> pd.DataFrame:
    return df.sort_values("FileDate", kind="stable").groupby(["Datum", *catcols]).first()


def add_date(df: pd.DataFrame, colname: str, format: str | None = None) -> pd.DataFrame:
    df["Datum"] = pd.to_datetime(df[colname], dayfirst=True, format=format, exact=format is not None)
    if colname != "Datum":
        df.drop(columns=colname, inplace=True)
    return df


def norm_df(df: pd.DataFrame, *, datecol: str, format: str | None = None) -> pd.DataFrame:
    add_date(df, datecol, format=format)
    return df


def shorten_bezname(bezname: str, *, soft: bool = False) -> str:
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


@dataclass(frozen=True, kw_only=True)
class EstiInfo:
    change_single: pd.DataFrame
    change_agg: pd.DataFrame
    change_agg_cum: pd.DataFrame
    change_agg_inner_cum: pd.DataFrame
    esti_len: int


def calc_esti(sariat: pd.Series, *, local_esti: bool = False) -> EstiInfo:
    # display(sariat)
    pltcol = sariat.name
    sariat = sariat.to_frame()
    sariat["age"] = (
        sariat.index.get_level_values("FileDate")
        - sariat.index.get_level_values("Datum").tz_localize("Europe/Vienna").tz_convert(UTC)
        - timedelta(7)
    )
    # sariat["weekno"] = (sariat.index.get_level_values("Datum") - sariat.index.unique("Datum").min()) // timedelta(7)
    sariat["i_age"] = sariat["age"] // timedelta(7)

    # display(sariat)

    sariat = pd.merge_asof(
        sariat,
        sariat[pltcol].rename("prev_report"),
        on="FileDate",
        by="Datum",
        allow_exact_matches=False,
    )
    sariat.set_index(["i_age", "Datum"], inplace=True, verify_integrity=True)
    # display(sariat)
    sariat["change_rel"] = (sariat[pltcol] / sariat["prev_report"]).shift(1)
    sariat["change_rel"] = sariat["change_rel"].where(np.isfinite(sariat["change_rel"]))
    sariat["change_rel_cum"] = sariat.groupby(["FileDate"])["change_rel"].transform(lambda s: s.cumprod()[::-1])
    # display(sariat)

    change_rel_r = sariat.groupby("i_age")["change_rel"].agg(["min", "max", "median"]).sort_index()
    esti_len = 10
    change_cum_r = change_rel_r[::-1].cumprod().sort_index().loc[:esti_len]

    change_cum_inner_r = sariat.groupby("i_age")["change_rel_cum"].agg(["min", "max", "median"]).sort_index()

    sariat.sort_index(inplace=True)

    if local_esti:
        # Reporting factors using “local” real-time data ("conservative estimate")
        # Equation (8) in Section 3.2 "Estimating reporting factors"from:
        #
        # Beesley LJ, Osthus D, Del Valle SY (2022)
        # Addressing delayed case reporting in infectious disease forecast modeling.
        # PLoS Comput Biol 18(6): e1010115.
        # https://doi.org/10.1371/journal.pcbi.1010115

        # Notation translation: t=date (week), d=i_age (lag)
        compl_after_weeks = 10  # K

        def calc_local_esti(date, i_age):
            fromdate = date - compl_after_weeks * timedelta(7)
            todate = date - (i_age + 1) * timedelta(7)
            daterng = pd.date_range(fromdate, todate, freq="W-MON")
            # print(f"{date=} {i_age=} {fromdate=} {todate=}")
            older = sariat.loc[pd.IndexSlice[i_age, fromdate:todate], pltcol]

            if older.sum() == 0 or len(older) != len(daterng):
                # print("short", older)
                return np.nan
            newidxs = list(zip((date - daterng) // timedelta(7), daterng, strict=True))
            # print(f"{newidxs=}")
            newer = sariat.loc[newidxs, pltcol]
            if len(newer) != len(daterng):
                return np.nan
            # display(older.rename("older"), newer.rename("newer"))
            return newer.sum() / older.sum()

        # sariat.set_index(["i_age", "Datum"], inplace=True, verify_integrity=True)
        # display(sariat)
        for i_age in range(compl_after_weeks):
            sariat.loc[i_age, "change_loc"] = (
                sariat.loc[i_age]
                .index.get_level_values("Datum")
                .map(lambda dt: calc_local_esti(dt, i_age))  # noqa: B023
            )

    return EstiInfo(
        change_single=sariat,
        change_agg=change_rel_r,
        change_agg_cum=change_cum_r,
        change_agg_inner_cum=change_cum_inner_r,
        esti_len=esti_len,
    )
