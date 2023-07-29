#!/usr/bin/env python3

"""Main code of the bot behind https://twitter.com/covidatTicker

Offers a commandline interface as well to compare specific AGES
data drops (see --help).
"""

import dataclasses
import json
import locale
import logging
import os
import sys
import traceback
import typing
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from itertools import repeat
from math import ceil
from pathlib import Path
from typing import Any, NamedTuple
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .covdata import AGES_TIME_FMT, SHORTNAME2_BY_BUNDESLAND, add_date, shorten_bezname
from .util import COLLECTROOT, Openable

try:
    from tweepy import Client  # type: ignore
except ModuleNotFoundError as exc:
    Client = None
    tweepy_error = exc
else:
    tweepy_error = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DayData:
    creationdate: pd.Timestamp
    bezdata: pd.DataFrame
    agdata: pd.DataFrame
    sumdead: int

    @property
    def last_date(self) -> pd.Timestamp:
        return typing.cast(pd.Timestamp, self.agdata.index.get_level_values("Datum")[-1])


@dataclass(frozen=True)
class DiffData:
    old: DayData
    new: DayData
    diff: pd.DataFrame

    @property
    def is_intraday(self) -> bool:
        return self.new.last_date.date() == self.old.last_date.date()


def tlen(s):
    # Extremely simplified (& over-estimating) tweet len calculation
    return sum(1 if ord(c) <= 0x10FF else 2 for c in s)


AG_IDX_COLS = ["Datum", "BundeslandID", "AltersgruppeID", "Geschlecht"]
AG_DIFF_IDX_COLS = [*AG_IDX_COLS, "Bezirk"]


def load_dead(fname: Openable) -> DayData:
    bezcols = {
        "Time": str,
        # "Bundesland": str,
        "GKZ": int,
        "Bezirk": str,
        # "AnzEinwohner": int,
        # "AnzahlFaelle": int,
        # "AnzahlFaelleSum": int,
        # "AnzahlFaelle7Tage": int,
        "AnzahlTotTaeglich": int,
        "AnzahlTotSum": int,
    }
    agcols = {
        "Time": str,
        "Bundesland": str,
        "BundeslandID": int,
        "AnzahlTot": int,
        "AltersgruppeID": int,
        "Geschlecht": str,
        "Altersgruppe": str,
        # "AnzahlGeheiltTaeglich": int,
        # "AnzahlGeheiltSum": int,
    }

    csvargs = {"engine": "c", "header": 0, "encoding": "utf-8", "sep": ";"}
    fname = Path(fname)
    if fname.is_dir():
        verdata = pd.read_csv(fname / "Version.csv", **csvargs)
        bez = pd.read_csv(
            fname / "CovidFaelle_Timeline_GKZ.csv",
            dtype=bezcols,
            usecols=list(bezcols.keys()),
            **csvargs,
        )
        ag = pd.read_csv(
            fname / "CovidFaelle_Altersgruppe.csv",
            dtype=agcols,
            usecols=list(agcols.keys()),
            **csvargs,
        )
    else:
        with ZipFile(fname) as zf:
            with zf.open("Version.csv") as verf:
                verdata = pd.read_csv(verf, **csvargs)
            with zf.open("CovidFaelle_Timeline_GKZ.csv") as bezf:
                bez = pd.read_csv(bezf, dtype=bezcols, usecols=list(bezcols.keys()), **csvargs)

            with zf.open("CovidFaelle_Altersgruppe.csv") as agf:
                ag = pd.read_csv(agf, dtype=agcols, usecols=list(agcols.keys()), **csvargs)
    creationdate = pd.to_datetime(verdata.iloc[0]["CreationDate"], dayfirst=True)

    add_date(bez, "Time", format=AGES_TIME_FMT)
    add_date(ag, "Time", format=AGES_TIME_FMT)

    bez["BundeslandID"] = bez["GKZ"] // 100
    bez.rename(columns={"AnzahlTotTaeglich": "AnzahlTot"}, errors="raise", inplace=True)

    ag.set_index(AG_IDX_COLS, inplace=True)
    ag.drop(10, level="BundeslandID", inplace=True)
    ag.rename(columns={"AnzahlTot": "AnzahlTotSum"}, errors="raise", inplace=True)
    ag["AnzahlTot"] = ag["AnzahlTotSum"].groupby(level=ag.index.names[1:]).diff()
    deadnan = np.isnan(ag["AnzahlTot"])
    ag.loc[deadnan, "AnzahlTot"] = ag.loc[deadnan, "AnzahlTotSum"]
    ag["AnzahlTot"] = ag["AnzahlTot"].astype(int)
    bez = bez.set_index(["Datum", "BundeslandID", "GKZ"]).query("BundeslandID != 10")

    sumdead = bez.loc[bez.index.get_level_values("Datum") == bez.index.get_level_values("Datum")[-1]][
        "AnzahlTotSum"
    ].sum()

    return DayData(creationdate=creationdate, bezdata=bez, agdata=ag, sumdead=sumdead)


def cmp_dead(old: DayData, new: DayData) -> DiffData:
    diff = new.agdata.copy()
    diff.drop(columns="AnzahlTotSum", inplace=True)
    diff["AnzahlTot"] = diff["AnzahlTot"].sub(old.agdata["AnzahlTot"], fill_value=0).astype(int)
    diff = diff.query("AnzahlTot != 0").copy()
    diffsum = diff["AnzahlTot"].sum()
    logger.debug("diffsum=%s diff=%s", diffsum, diff)
    diff["Bezirk"] = None
    diff.set_index("Bezirk", append=True, inplace=True)
    assert diff.index.names == AG_DIFF_IDX_COLS, f"{diff.index.names=} != {AG_DIFF_IDX_COLS=}"

    diffdt_bez = new.bezdata.copy()
    diffdt_bez["AnzahlTot"] = diffdt_bez["AnzahlTot"].sub(old.bezdata["AnzahlTot"], fill_value=0).astype(int)
    diffdt_bez = diffdt_bez.query("AnzahlTot != 0")
    logger.debug("diffdt_bez=%s", diffdt_bez)
    bezopen = diffdt_bez.copy()

    # Find Bezirk if possible
    sum_ag_open = 0
    origdiff = diff.copy()
    for row in origdiff.itertuples():
        bezkey = row[0][0:2]
        try:
            bezmatch = diffdt_bez.loc[bezkey]
        except KeyError:
            bezmatch = ()
        if (
            len(bezmatch) > 1
            and bezmatch["AnzahlTot"].sum() == row.AnzahlTot
            and len(origdiff.loc[bezkey]) == 1  # sum match not enough (neg. N)
            or len(bezmatch) == 1
        ):
            diff.drop(row[0], inplace=True)
            for bezkeytail, loc, anzTot in bezmatch[["Bezirk", "AnzahlTot"]].itertuples():
                key = row.Index[:-1] + (loc,)
                if len(bezmatch) != 1:
                    row = row._replace(AnzahlTot=anzTot)
                diff.loc[key] = row[1:]
                bezopen.loc[(*bezkey, bezkeytail), "AnzahlTot"] -= row.AnzahlTot
        else:
            sum_ag_open += row.AnzahlTot
    diff.index.set_names(origdiff.index.names, inplace=True)  # Workaround Pandas v2.0 bug
    left_open = bezopen["AnzahlTot"] != 0
    if left_open.any():
        sum_open = bezopen["AnzahlTot"].sum()
        logger.log(
            logging.INFO if sum_ag_open == sum_open else logging.WARN,
            "%d Bezirk-entries with %d deaths were not used/overused with %d deaths from age data not matched",
            left_open.sum(),
            sum_open,
            sum_ag_open,
        )
        logger.debug("Open entries: %s", bezopen.loc[left_open])
        if sum_ag_open != sum_open:
            raise ValueError("Inconsistent data bez/ag, see previous log message.")
    assert not (diff["AnzahlTot"] == 0).any()
    newdiffsum = diff["AnzahlTot"].sum()
    assert newdiffsum == diffsum, f"{diffsum=} {newdiffsum=}"
    return DiffData(old, new, diff.sort_index())


def reldate(ddate: pd.Timestamp, tod: pd.Timestamp) -> str:
    return (
        ddate.strftime("%a").removesuffix(".")
        if (tod - ddate).days < 7
        else ddate.strftime("%d.%m")
        if ddate.year == tod.year
        else ddate.strftime("%d.%m.%Y")
    )


def fmt_range(first: Any, second: Any) -> str:
    return f"{first}-{second}" if first != second else first


def fmt_dt_range(first: pd.Timestamp, second: pd.Timestamp, tod: pd.Timestamp):
    return fmt_range(reldate(first, tod), reldate(second, tod))


@dataclass(frozen=True)
class CorrInfo:
    n_corr: int
    mindate: pd.Timestamp
    maxdate: pd.Timestamp

    @staticmethod
    def merge(c1: "CorrInfo", c2: "CorrInfo") -> "CorrInfo":
        if c1.n_corr == 0:
            return c2
        if c2.n_corr == 0:
            return c1
        return CorrInfo(
            c1.n_corr + c2.n_corr,
            min(c1.mindate, c2.mindate),
            max(c1.maxdate, c2.maxdate),
        )


def split_multisum_entries(diffentries: pd.DataFrame) -> pd.DataFrame:
    origsum = diffentries["AnzahlTot"].sum()
    orig = diffentries
    diffentries = diffentries.copy()
    extra: list[pd.DataFrame] = [diffentries]

    def corr0(sel: pd.Series, dir: int) -> None:
        if not sel.any():
            return
        diffentries.loc[sel, "AnzahlTot"] -= dir
        corrs = diffentries.loc[sel].copy()
        corrs["AnzahlTot"] = dir
        extra.append(corrs)

    while True:
        gt1 = diffentries["AnzahlTot"] > 1
        ltm1 = diffentries["AnzahlTot"] < -1
        if not gt1.any() and not ltm1.any():
            break
        corr0(gt1, 1)
        corr0(ltm1, -1)

    diffentries = pd.concat(extra)  # .sort_index()
    diffentries["id"] = np.arange(len(diffentries), dtype=int)
    diffentries.set_index("id", append=True, inplace=True)
    diffentries.sort_index(inplace=True)  # Sort after making index unique
    assert len(diffentries) >= len(orig)
    assert origsum == diffentries["AnzahlTot"].sum(), f"{origsum} != {diffentries['AnzahlTot'].sum()}"
    return diffentries


class SummarySplit(NamedTuple):
    diff: pd.DataFrame
    summarydiff: pd.DataFrame


def split_summary(diff: pd.DataFrame, summarizeuntil: pd.Timestamp, *, keep_single_entry=True) -> SummarySplit:
    dts = diff.index.to_frame()["Datum"]
    summarized_sel = dts <= summarizeuntil
    summary = diff.loc[summarized_sel].reset_index("Datum").copy()
    if "loc" not in summary.columns:
        summary["loc"] = None

    # & ~diff["Altersgruppe"].isin(["<5", "5-14"])
    summary = summary.groupby("AltersgruppeID").agg(
        AnzahlTot=("AnzahlTot", "sum"),
        mindate=("Datum", "min"),
        maxdate=("Datum", "max"),
        first_bl=("Bundesland", "first"),
        n_bl=("Bundesland", lambda xs: xs.nunique(dropna=False)),
        first_loc=("loc", "first"),
        n_loc=("loc", lambda xs: xs.nunique(dropna=False)),
        Altersgruppe=("Altersgruppe", "first"),
    )
    summary["uloc"] = None
    uq_loc = summary["n_loc"] == 1
    if uq_loc.any():
        summary.loc[uq_loc, "uloc"] = summary.loc[uq_loc, "first_loc"].map(
            lambda xs: shorten_bezname(xs, soft=True), na_action="ignore"
        )
    uq_bl = summary["n_bl"] == 1
    if uq_bl.any():
        summary.loc[pd.isna(summary["uloc"]) & uq_bl, "uloc"] = summary.loc[
            pd.isna(summary["uloc"]) & uq_bl, "first_bl"
        ].map(SHORTNAME2_BY_BUNDESLAND, na_action="ignore")
    # summary.columns = summary.columns.droplevel(1)
    if keep_single_entry:
        unsummarize_sel = summary["AnzahlTot"].isin([-1, 1])
        result = SummarySplit(diff.loc[~summarized_sel | unsummarize_sel], summary[~unsummarize_sel])
    else:
        result = SummarySplit(diff.loc[~summarized_sel], summary)
    assert result.diff["AnzahlTot"].sum() + result.summarydiff["AnzahlTot"].sum() == diff["AnzahlTot"].sum()
    return result


def format_summary_rows(summary: pd.DataFrame, tod: pd.Timestamp) -> list[str]:
    return [
        f"{row.AnzahlTot:+.0f} {fmt_dt_range(row.mindate, row.maxdate, tod)}"
        f" {row.Altersgruppe}" + (f" {row.uloc}" if "uloc" in dir(row) and not pd.isna(row.uloc) else "") + "\n"
        for row in summary.itertuples(index=False)
    ]


CURRENT_PERIOD_NDAYS = 110  # Before this, we try to avoid e.g. any corrections spilling over
RECENT_NDAYS = 21  # In the last N days, we want to see all details.


def without_zerosum_corr(diffentries: pd.DataFrame, mincorrdate: pd.Timestamp) -> tuple[pd.DataFrame, CorrInfo]:
    # For all zero-csums, drop the entries of the same age group before that

    diffentries = split_multisum_entries(diffentries)  # diffentries.copy() #
    csum = diffentries.groupby("AltersgruppeID")["AnzahlTot"].cumsum()
    origsum = diffentries["AnzahlTot"].sum()

    orig_idx_names = diffentries.index.names

    deluntil = (
        diffentries.loc[(csum == 0) & (diffentries.index.get_level_values("Datum") <= mincorrdate)]
        .reset_index()
        .groupby("AltersgruppeID")
        .last()
        .reset_index()
        .set_index(orig_idx_names)
        .index
    )

    mindate = None
    maxdate = diffentries.iloc[0].name[0] if len(diffentries) > 0 else None
    n_corr = 0
    for idx in deluntil:
        dropped = diffentries.loc[:idx]
        dropped = dropped.loc[
            dropped.index.get_level_values("AltersgruppeID") == idx[deluntil.names.index("AltersgruppeID")]
        ]
        n_corr += len(dropped)
        droppedmindate = dropped.index[0][0]
        mindate = droppedmindate if mindate is None else min(mindate, droppedmindate)
        maxdate = max(maxdate, dropped.index[-1][0])
        diffentries.drop(index=dropped.index, inplace=True)

    assert origsum == diffentries["AnzahlTot"].sum(), f"{origsum} != {diffentries['AnzahlTot'].sum()} (1)"
    # Regroup split entries
    agg = {k: "first" for k in diffentries.columns if k != "AnzahlTot"} | {"AnzahlTot": np.sum}
    diffentries = diffentries.groupby(level=AG_DIFF_IDX_COLS, dropna=False).agg(agg)
    assert origsum == diffentries["AnzahlTot"].sum(), f"{origsum} != {diffentries['AnzahlTot'].sum()} (2)"
    return (diffentries, CorrInfo(n_corr=n_corr, mindate=mindate, maxdate=maxdate))


def format_dead(diff: DiffData) -> str:
    if len(diff.diff) == 0:
        return ""
    tod = diff.new.creationdate
    diffentries = diff.diff.query("BundeslandID != 10").copy()
    hasbezirk = ~pd.isna(diffentries.index.get_level_values("Bezirk"))
    diffentries.loc[hasbezirk, "loc"] = diffentries.index.get_level_values("Bezirk")[hasbezirk]
    diffentries.loc[hasbezirk, "loc"] = diffentries.loc[hasbezirk, "loc"].map(lambda xs: shorten_bezname(xs, soft=True))
    diffentries.loc[~hasbezirk, "loc"] = diffentries.loc[~hasbezirk, "Bundesland"].map(SHORTNAME2_BY_BUNDESLAND)
    diffentries.loc[~hasbezirk, "loc"] = diffentries.loc[~hasbezirk, "loc"].str.removesuffix(".")

    diffentries, corrinfo = without_zerosum_corr(diffentries, tod - timedelta(RECENT_NDAYS + 7))
    logger.debug("cleaned diffentries:\n%s", diffentries)

    result = []

    if len(diffentries) > 5:
        diffentries, summary = split_summary(diffentries, tod - timedelta(CURRENT_PERIOD_NDAYS))
        result += format_summary_rows(summary, tod)
        diffentries, corrinfo2 = without_zerosum_corr(diffentries, tod - timedelta(RECENT_NDAYS + 7))
        corrinfo = CorrInfo.merge(corrinfo, corrinfo2)

    def fmt_row(row):
        rowout = []
        ddate, sex = row[0][0], row[0][3]
        ag = row.Altersgruppe
        for _ in range(max(1, row.AnzahlTot)):
            if row.AnzahlTot <= 0:
                rowout.extend((str(row.AnzahlTot), " "))
            rowout.extend(
                (
                    " ".join(
                        (
                            reldate(ddate, tod),
                            sex,
                            ag,
                            row.loc,
                        )
                    ),
                    "\n",
                )
            )
        return rowout

    for row in diffentries.itertuples():
        result += fmt_row(row)

    if corrinfo.n_corr != 0:
        if result:
            result.append("Weiters ")
        result.append(f"{corrinfo.n_corr} Ummeldungen, {fmt_dt_range(corrinfo.mindate, corrinfo.maxdate, tod)}\n")

    return "".join(result)


EMOJI_BY_AGE = {
    k: {"M": m, "W": w}
    for k, (m, w) in {
        "<5": ("👶", "👶"),
        "5-14": ("👦", "👧"),
        "15-24": ("👦", "👧"),
        "25-34": ("👨", "👩"),
        "35-44": ("👨", "👩"),
        "45-54": ("👨", "👩"),
        "55-64": ("👨‍🦳", "👩‍🦳"),
        "65-74": ("👨‍🦳", "👩‍🦳"),
        "75-84": ("👴", "👵"),
        ">84": ("👴", "👵"),
    }.items()
}


def diff_to_emojis(diffentries: pd.DataFrame) -> Iterable[str]:
    sex_idx = diffentries.index.names.index("Geschlecht")
    for row in diffentries.itertuples():
        if row.AnzahlTot <= 0:
            raise ValueError(f"Cannot display row with negative count as emojis: {row}")
        sex = row[0][sex_idx]
        ag = row.Altersgruppe
        yield from repeat(EMOJI_BY_AGE[ag][sex], row.AnzahlTot)


MAX_TWEET_LEN = 278  # Two chars space for errors


def format_dead_tweets(diff: DiffData, silent_intraday: bool = False) -> list[str]:
    rows_raw = format_dead(diff)
    datestamp = diff.new.creationdate.strftime("%a %d.%m.%Y" + (" %H:%M" if diff.is_intraday else ""))
    allsum = diff.new.sumdead
    allsum_s = format(allsum, ",").replace(",", ".")
    epilog = f"{allsum_s} COVID-Tote wurden bisher in Österreich gezählt."
    if not rows_raw:
        if silent_intraday and diff.is_intraday:
            return []
        return [
            f"Mit Datenstand {datestamp} wurden seit der letzten Meldung"
            f" keine Änderungen bei #Covid19at-Todesfällen erfasst.\n\n{epilog}"
        ]
    rows = rows_raw.strip().split("\n")

    header = f"#COVID19at-Tote, gemeldet {datestamp}:\n\n"

    single = header + "\n".join(rows)
    if tlen(single) <= MAX_TWEET_LEN:
        withextra = single + "\n\n" + epilog
        return [withextra] if tlen(withextra) <= MAX_TWEET_LEN else [single]

    def split_rows(*, basefooterlen, headerlen, contheaderlen):
        curtweet_rows = []
        tweet_contents = [curtweet_rows]
        curlen = headerlen + basefooterlen + tlen("1")
        for row in rows:
            rowlen = tlen(row) + tlen("\n")
            if curlen + rowlen > MAX_TWEET_LEN:
                if len(tweet_contents) == 1:
                    headerlen = contheaderlen
                curtweet_rows = []
                tweet_contents.append(curtweet_rows)
                curlen = headerlen + basefooterlen + tlen(str(len(tweet_contents) + 1))
            curtweet_rows.append(row)
            curlen += rowlen
        return tweet_contents

    basefooterlen = 0
    nextbasefooterlen = tlen("/N")
    contheader = "… " + header
    while basefooterlen != nextbasefooterlen:
        basefooterlen = nextbasefooterlen
        tweet_contents = split_rows(
            basefooterlen=basefooterlen,
            headerlen=len(header),
            contheaderlen=tlen(contheader),
        )
        basefooter = f"/{len(tweet_contents)}"
        nextbasefooterlen = tlen(basefooter)

    tweets: list[str] = []
    for content in tweet_contents:

        def mktweet():
            return header + "\n".join(content) + f"\n{len(tweets) + 1}/{len(tweet_contents)}"

        tweet = mktweet()
        if content is tweet_contents[-1]:
            content += ["", epilog]
            extratweet = mktweet()
            if tlen(extratweet) <= MAX_TWEET_LEN:
                tweet = extratweet
        assert tlen(tweet) <= MAX_TWEET_LEN, f"{tlen(tweet)=}"
        if not tweets:
            header = contheader
        tweets.append(tweet)
    return tweets


def to_diff(old: Openable, new: Openable):
    return cmp_dead(load_dead(old), load_dead(new))


def join_tweets(tweets: Iterable[str]):
    return "\n\n---\n\n".join(tweets)


def print_cmp_one(old: Openable, new: Openable):
    diff = to_diff(old, new)
    print(join_tweets(format_dead_tweets(diff) + format_weekstat_tweets(diff)))
    print(flush=True)


def run_test(last_n: int) -> None:
    paths = COLLECTROOT.glob("2*/*_*_orig_csv_ages")
    botio = InMemoryBotStateIo()
    for path in tuple(sorted(paths))[-last_n:]:
        try:
            print(f"run_bot({botio.persisted_state}, {path})", flush=True)
            run_bot(botio, path)
        except (FileNotFoundError, KeyError):
            traceback.print_exc()
            input("...")


def format_weekstat_tweets(weekdiff: DiffData) -> list[str]:
    weektstamp = weekdiff.new.creationdate.strftime("KW %V bis %d.%m.%Y")
    header = f"Diese Woche, {weektstamp}, gemeldete #COVID19at-Tote:\n\n"
    fulldiffentries = weekdiff.diff.query("BundeslandID != 10")
    # Don't spill over old corrections into the last few days
    # (we redo the zerosum-corr after splitting of old corrections)
    fulldiffentries, _ = without_zerosum_corr(
        fulldiffentries, mincorrdate=weekdiff.old.creationdate - timedelta(RECENT_NDAYS)
    )
    diffentries, oldsummary = split_summary(
        fulldiffentries,
        weekdiff.old.creationdate - timedelta(CURRENT_PERIOD_NDAYS - 1),
        keep_single_entry=False,
    )
    oldsum = oldsummary["AnzahlTot"].sum()
    logger.debug("weekstat: diffentries scrubbed from %s old:\n%s", oldsum, diffentries)
    if oldsum != 0:
        oldnote = f"({oldsum:+.0f} bis {reldate(oldsummary['maxdate'].max(), weekdiff.new.creationdate)})\n"
        if oldsum < 0 or (oldsum > 0 and tlen(oldnote) < oldsum * 2):
            header += oldnote
        else:
            # We have a nonnegative number of deaths that fall before our cutoff
            # date, but the note about it would be longer than just displaying
            # the emojis, so just pretend they fall after our cutoff.
            diffentries = fulldiffentries
    diffentries, _ = without_zerosum_corr(diffentries, mincorrdate=weekdiff.new.creationdate)
    logger.debug("weekstat: diffentries with zerosum_corr:\n%s", diffentries)
    neg_sel = diffentries["AnzahlTot"] < 0
    if neg_sel.any():
        negsum = diffentries.loc[neg_sel]["AnzahlTot"].sum()
        fromdate = diffentries[neg_sel].index.get_level_values("Datum").min()
        assert negsum < 0
        header += f"({negsum:+.0f} seit {reldate(fromdate, weekdiff.new.creationdate)})\n"
        diffentries = diffentries.loc[~neg_sel]

    # Keep this in a list[str], because emojis count as only 1 in tweets
    content: list[str] = list(diff_to_emojis(diffentries))
    if tlen(header) + len(content) * 2 <= MAX_TWEET_LEN:
        return [header + "".join(content)]

    firstcnt = (MAX_TWEET_LEN - tlen(header) - tlen("1/NN")) // 2
    remcnt = len(content) - firstcnt
    per_chunk_cnt = (MAX_TWEET_LEN - tlen("ii/NN")) // 2
    tweetcnt = 1 + ceil(remcnt / per_chunk_cnt)
    if tweetcnt > 99:
        raise ValueError(f"Week summary too large: {tweetcnt=}")
    tweets = [header + "".join(content[:firstcnt]) + f"1/{tweetcnt}"]
    for i, chunkstart in enumerate(range(firstcnt, len(content), per_chunk_cnt), 2):
        tweets.append("".join(content[chunkstart : chunkstart + per_chunk_cnt]) + f"{i}/{tweetcnt}")
    assert len(tweets) == tweetcnt, f"{len(tweets)=} == {tweetcnt=})"
    return tweets


@dataclass(frozen=True)
class BotState:
    lastfile: str | None
    replyto: str | None = None
    lastfile_weekly: str | None = None
    brokenreplyto: str | None = None


def next_bot_state(newfile: str, botstate: BotState) -> tuple[list[str], BotState]:
    if botstate is None or not botstate.lastfile:
        return ([], BotState(lastfile=newfile))
    if newfile == botstate.lastfile:
        logger.info("Nothing to do, file already processed.")
        return ([], botstate)
    newdata = load_dead(newfile)
    diffdata = cmp_dead(load_dead(botstate.lastfile), newdata)
    tweets = format_dead_tweets(diffdata, silent_intraday=True)

    if newdata.creationdate.weekday() == 6:
        if botstate.lastfile_weekly:
            lastweek = load_dead(botstate.lastfile_weekly)
            if newdata.creationdate - lastweek.creationdate >= timedelta(6):
                week_diffdata = cmp_dead(lastweek, newdata)
                tweets += format_weekstat_tweets(week_diffdata)
                botstate = dataclasses.replace(botstate, lastfile_weekly=newfile)
        else:
            botstate = dataclasses.replace(botstate, lastfile_weekly=newfile)
    if not tweets:
        logger.info("Skipping intraday comparison with no relevant output")
        # Also will not update lastfile
    return tweets, dataclasses.replace(botstate, lastfile=newfile)


class BotStateIo(ABC):
    @abstractmethod
    def read_bot_state(self) -> BotState | None:
        pass

    @abstractmethod
    def write_botstate(self, state: BotState, only_create=False) -> None:
        pass

    @abstractmethod
    def get_twitter_client(self) -> Client:
        pass

    @abstractmethod
    def dump_result(self, dumpinfo: str) -> None:
        pass


class InMemoryBotStateIo(BotStateIo):
    def __init__(self, botstate: BotState | None = None):
        self.persisted_state = botstate

    def read_bot_state(self) -> BotState | None:
        return self.persisted_state

    def write_botstate(self, state: BotState, only_create=False) -> None:
        if only_create and self.persisted_state is not None:
            raise FileExistsError("State already exists " + repr(self.persisted_state))
        self.persisted_state = state

    def get_twitter_client(self) -> Client:
        raise NotImplementedError("Cannot actually tweet with InMemoryBotStateIo")

    def dump_result(self, dumpinfo: str) -> None:
        print(dumpinfo)


class ProdBotStateIo(BotStateIo):
    _STATEFILENAME = "botstate.json"
    _OUTFILENAME = "botout.txt"

    def read_bot_state(self) -> BotState | None:
        try:
            statef = open(self._STATEFILENAME, "rb")
        except FileNotFoundError as exc:
            logger.warning("state file %s not present %s", self._STATEFILENAME, exc)
            return None
        with statef:
            rawstate = json.load(statef)
        return BotState(**rawstate)

    def write_botstate(self, state: BotState, only_create=False) -> None:
        statedata = dataclasses.asdict(state)
        with open(self._STATEFILENAME, "x" if only_create else "w", encoding="utf-8") as statef:
            json.dump(statedata, statef, indent=2)

    def get_twitter_client(self) -> Client:
        creds = {}
        for cred_part in (
            "consumer_key",
            "consumer_secret",
            "access_token",
            "access_token_secret",
        ):
            key = f"COVAT_TWITTER_{cred_part.upper()}"
            part = os.environ.get(key)
            if not part:
                raise ValueError("Missing credential part: " + key)
            creds[cred_part] = part
        return Client(**creds)

    def dump_result(self, dumpinfo: str) -> None:
        OUTFILENAME = "botout.txt"
        with open(OUTFILENAME, "a", encoding="utf-8") as outf:
            outf.write(dumpinfo)


def run_bot(botio: BotStateIo, newfile: Openable) -> None:
    newfile = str(newfile)
    botstate = botio.read_bot_state()
    prevfile = botstate.lastfile or "(None)" if botstate else "(No botstate)"
    tweets, botstate = next_bot_state(newfile, botstate)
    if tweets:
        results = join_tweets(tweets)

        botio.dump_result(f"\n\n==== {prevfile} {newfile}\n\n{results}\n")

        replyto = botstate.replyto
        if replyto and botstate.brokenreplyto is None:
            client = botio.get_twitter_client()
            try:
                if len(tweets) > 7:
                    raise ValueError(f"Too many tweets, balking: {len(tweets)}")
                for tweet in tweets:
                    resp = client.create_tweet(in_reply_to_tweet_id=replyto, text=tweet)
                    replyto = resp.data["id"]
                    logger.info("Created tweet %s", replyto)
            except:
                # Error occurred during tweet (even KeyboardInterrupt) =>
                # Stop tweeting until manual intervention
                botio.write_botstate(dataclasses.replace(botstate, brokenreplyto=replyto))
                raise
            botstate = dataclasses.replace(botstate, replyto=replyto)
    botio.write_botstate(botstate)


def main() -> None:
    locale.setlocale(locale.LC_ALL, "de_AT.UTF-8")

    parser = ArgumentParser()
    parser.add_argument("--logfile")
    parser.add_argument("--loglevel")
    subparsers = parser.add_subparsers(required=True, dest="cmd")
    cmp = subparsers.add_parser("cmp")
    cmp.add_argument("old")
    cmp.add_argument("new")
    testp = subparsers.add_parser("test")
    testp.add_argument("--count", type=int, default=2)

    runbotp = subparsers.add_parser("runbot")
    runbotp.add_argument("new")

    args = parser.parse_args()

    logging.basicConfig(
        filename=args.logfile,
        level=args.loglevel.upper() if args.loglevel else args.loglevel,
    )

    if args.cmd == "cmp":
        print_cmp_one(args.old, args.new)
    elif args.cmd == "test":
        run_test(args.count)
    elif args.cmd == "runbot":
        if tweepy_error is not None:
            sys.exit(f"runbot command unailable: {type(tweepy_error).__qualname__}: {tweepy_error}")
        run_bot(ProdBotStateIo(), args.new)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
