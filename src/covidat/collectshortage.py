#!/usr/bin/env python3

import itertools
import sys
import traceback
from datetime import date, datetime
from pathlib import Path
import csv
import html
import re
from typing import Set
import warnings
import pandas as pd
import logging
from . import util

logger = logging.getLogger(__name__)

HWS_RE = re.compile(r"[ \t\u00A0]+")
RESCOUNT_RE = re.compile(r"\b(\d+) Ergebnis(?:se)? gefunden\b")


def match_number(pat: re.Pattern, text: str) -> int:
    match = pat.search(text)
    if not match:
        raise KeyError(f"Could not find a match for: {pat.pattern}")
    return int(match.group(1).replace(".", ""))


def collectshortage(dirname, outname):
    with open(outname, "w", encoding="utf-8", newline="\n") as outfile:
        writer = csv.DictWriter(
            outfile,
            ("Datum", "NLimitedMeds"),
            delimiter=";",
        )
        writer.writeheader()
        for fname in sorted(Path(dirname).glob("adf_*_*.task-flow")):
            with open(fname, encoding="utf-8") as f:
                fdata = HWS_RE.sub(" ", html.unescape(f.read()))
            fdate = (
                datetime.strptime(fname.stem.split("_", 1)[1], "%Y%m%d_%H%M%S")
                .date()
                .isoformat()
            )
            try:
                writer.writerow(
                    {
                        "Datum": fdate,
                        "NLimitedMeds": match_number(RESCOUNT_RE, fdata),
                    }
                )
            except KeyError as e:
                raise ValueError(f"Error in {fname}: {e}") from e


def norm_name(name: pd.Series):
    return (
        name.replace(r"\b(\d|,)+\b", "0", regex=True)
        .replace(r"\b\d+([mg])", r"0 \1", regex=True)
        .replace(r"Filmtabletten?", "Tabletten", regex=True)
        .replace(r"(\s|[-\"])+", " ", regex=True)
        .str.strip()
    )


def load_azr():
    cachefile = util.COLLECTROOT / "medshort/ASP-Register.pkl"
    try:
        azr = pd.read_pickle(cachefile)
    except FileNotFoundError:
        logger.info("Regenerating ASP-Register cache...")
        srcs = []
        aspdir = util.DATAROOT / "basg-medicineshortage"
        for srcname in aspdir.glob("ASP-Register_2*.xlsx"):
            srcs.append(pd.read_excel(srcname, header=0))
        azr = pd.concat(srcs)
        azr.to_pickle(cachefile)
        logger.info("Regenerated cache at %s", cachefile)
    azr["Zulassungsnummer"] = azr["Zulassungsnummer"].str.strip()
    azr.drop_duplicates("Zulassungsnummer", keep="last", inplace=True)
    msk = azr["Zulassungsnummer"].str.match("EU/.+[-,]")
    azr0 = azr.loc[msk].copy()
    azr0["Zulassungsnummer"] = azr0["Zulassungsnummer"].str.replace(
        "[-,][^/]+$", "", regex=True, n=1
    )
    azr["nkey"] = norm_name(azr["Name"])
    azr_m = pd.read_csv(
        util.DATAROOT / "basg-medicineshortage/ASP-Missing.csv", sep=";"
    )
    azr = pd.concat([azr, azr0, azr_m]).reset_index(drop=True)
    return azr


def agg_status(s: pd.Series):
    return (
        "verfügbar"
        if (s == "verfügbar").all()
        else "teilweise verfügbar"
        if "verfügbar" in s.values
        else next((p for p in cat_prio if p in s.values))
    )


cat_prio = [
    "nicht verfügbar",
    "eingeschränkt verfügbar",
    "verfügbar gemäß §4 (1)",
    "teilweise verfügbar",
    "verfügbar",
][::-1]


def load_veasp_xml(fname, azr: pd.DataFrame, only_statagg=False) -> pd.DataFrame:
    veasp = pd.read_xml(fname, parser="etree", xpath="./Packungen/Packung")
    veasp.rename(
        columns={"Bezeichnung_Arzneispezialitaet": "Name"},
        inplace=True,
        errors="raise",
    )
    veasp["Zulassungsnummer"] = veasp["Zulassungsnummer"].str.strip()
    veasp = veasp.merge(
        azr[["Zulassungsnummer", "Verwendung"]],
        "left",
        on="Zulassungsnummer",
    )
    msk = pd.isna(veasp["Verwendung"]) & veasp["Zulassungsnummer"].str.match(
        "EU/.+[-,]"
    )
    veasp.loc[msk, "Zulassungsnummer"] = veasp.loc[msk, "Zulassungsnummer"].str.replace(
        "[-,][^/]+$", "", regex=True, n=1
    )
    veasp = veasp.merge(
        azr[["Zulassungsnummer", "Verwendung"]],
        "left",
        on="Zulassungsnummer",
        suffixes=("", "_y"),
    )
    # display(veasp)
    veasp["Status"].replace("Nicht verfügbar", "nicht verfügbar", inplace=True)
    veasp["Verwendung"] = veasp["Verwendung"].combine_first(veasp["Verwendung_y"])
    naveasp = veasp[pd.isna(veasp["Verwendung"])]
    if len(naveasp) > 0:
        raise ValueError(
            str(fname)
            + ": Not found in AZR: "
            + naveasp[["Zulassungsnummer", "Name"]].to_csv(sep=";", index=False)
        )
    # veasp.sort_values(["Avail_c"], kind="stable", inplace=True)
    # veasp.drop(columns=["Verwendung_y"], inplace=True)
    try:
        agg = {
            "Name": "first",
            "Status": agg_status,
        }
        if not only_statagg:
            agg |= {
                "Grund": lambda g: " / ".join(g.unique()),
                "Melder": lambda w: " / ".join(w.unique()),
                "Melder": lambda w: " / ".join(w.unique()),
                "Zulassungsinhaber": lambda w: " / ".join(w.unique()),
                "Wirkstoffe": lambda w: pd.unique(", ".join(w.unique()).split(", ")),
                "Datum_Meldung":lambda s: pd.to_datetime(s, format="%Y-%m-%d").min(),
                "Datum_letzte_Aenderung": lambda s: pd.to_datetime(s, format="%Y-%m-%d").max(),
                "Beginn_Vertriebseinschraenkung": lambda s: pd.to_datetime(s, format="%Y-%m-%d").min()
            }
        return (
            veasp.groupby(["Zulassungsnummer", "Verwendung"]).agg(agg)
        ).reset_index()
    except StopIteration:
        raise ValueError(f"{fname}: Bad status: {veasp['Status'].unique()}")


def processfile(dts: Set[date], fname: Path, azr: pd.DataFrame, writer: csv.DictWriter):
    fdate = datetime.strptime(
        fname.stem.split("_", 1)[1].split(".", 1)[0], "%Y%m%d_%H%M%S"
    ).date()
    if fdate in dts:
        return
    if fname.name.endswith(".xlsx"):
        source_format = "xlsx"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            data = pd.read_excel(fname, header=0)
        data.rename(
            columns={n: n.strip() for n in data.columns},
            inplace=True,
            errors="raise",
        )
        for c, dt in data.dtypes.items():
            if dt == "object":
                # print("s", c, dt)
                data[c] = data[c].str.strip()
        data["Status"] = data["Status"].str.rstrip("*")
        # data = data.query("Status != 'verfügbar'")
        # print(data.dtypes)
        data.loc[
            data["PZN wieder verfügbarer Packungen"].astype(bool)
            & ~pd.isna(data["PZN wieder verfügbarer Packungen"])
            & (data["Status"] != "verfügbar"),
            "Status",
        ] = "teilweise verfügbar"
        data.loc[
            (
                ~data["PZN eingeschränkt verfügbarer Packungen"].astype(bool)
                | pd.isna(data["PZN eingeschränkt verfügbarer Packungen"])
            )
            & (data["Status"] == "eingeschränkt verfügbar"),
            "Status",
        ] = "teilweise verfügbar"
    else:
        source_format = "xml"
        data = load_veasp_xml(fname, azr, only_statagg=True)

    if False:  # Group by name
        dkey = norm_name(data["Name"])
        try:
            data = data.groupby([dkey, data["Verwendung"]]).agg({"Status": agg_status})
        except StopIteration:
            raise ValueError(f"{fname}: Bad status: {data['Status'].unique()}")
        # print(fname)
        # print("\n".join(sorted(dkey.unique())))
        # input("...")
    for usage, data_u in data.groupby("Verwendung"):
        for avail, data_uv in data_u.groupby("Status"):
            # if avail == 'verfügbar':
            #    print(fname)
            #    print(data_uv)
            #    input("...")
            writer.writerow(
                {
                    "Datum": fdate.isoformat(),
                    "Usage": usage,
                    "Availability": avail,
                    "N": len(data_uv),
                    "SourceFormat": source_format,
                }
            )
    dts.add(fdate)


def collectshortage_ex(dirname, outname):
    azr = load_azr()
    dts = set()
    with open(outname, "w", encoding="utf-8", newline="\n") as outfile:
        writer = csv.DictWriter(
            outfile,
            ("Datum", "Usage", "Availability", "N", "SourceFormat"),
            delimiter=";",
        )
        writer.writeheader()
        searchpath = Path(dirname)
        for fname in sorted(
            itertools.chain(
                searchpath.glob("VertriebseinschraenkungenASP_*_*.xml"),
                searchpath.glob("Vertriebseinschraenkungen_*_*.xlsx"),
            )
        ):
            try:
                processfile(dts, fname, azr, writer)
            except Exception as e:
                traceback.print_exc()
                print(
                    "The previous exception occured during processing of",
                    fname,
                    file=sys.stderr,
                    flush=True,
                )


def main():
    collectdir = util.COLLECTROOT / "medshort"
    collectdir.mkdir(parents=True, exist_ok=True)
    datadir = util.DATAROOT / "basg-medicineshortage"

    collectshortage(datadir, collectdir / "medshort.csv")
    collectshortage_ex(datadir, collectdir / "medshort_ex.csv")


if __name__ == "__main__":
    main()
