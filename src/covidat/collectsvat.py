import re
from . import util
import pandas as pd
import numpy as np


def _ks_colnames(sfx):
    return [c + sfx for c in KS_BASECOLS]


KS_BASECOLS = ["insured", "cases", "cases_p1000", "active_end", "active_end_p1000"]
KS_COLNAMES = (
    [
        "id",
        "insurer",
    ]
    + _ks_colnames("-ArbeiterInnen")
    + _ks_colnames("-Angestellte")
)
KS_FNAME_RE = re.compile("Mb_(\d\d)(\d\d)")

SVDIRNAME = "sozialversicherung-monatsberichte"


def load_ks(pth):
    m = KS_FNAME_RE.fullmatch(pth.stem)
    if not m:
        raise ValueError("Unexpected filename stem: '" + pth.stem + "' in: " + str(pth))
    year = 2000 + int(m.group(1))
    month = int(m.group(2))

    ks = pd.read_excel(
        pth, sheet_name="Tab16", skiprows=9, header=None, names=KS_COLNAMES,
    )

    ks.dropna(thresh=3, inplace=True)
    if pd.isna(ks["insurer"].iloc[0]):
        raise ValueError("Missing insurer")
    ks["date"] = (ks["id"] % 2).map(
        lambda odd: pd.Period(year=year if odd else year - 1, month=month, freq="M")
    )
    return ks


def collect_ks():
    ks = pd.concat(
        [
            load_ks(pth)
            for pth in (util.DATAROOT / SVDIRNAME).glob(
                "Mb_????.xls*"
            )
        ]
    )
    ks["insurer"].replace(
        "I n s g e s a m t|insgesamt|ASVG-Krankenkassen",
        "Insgesamt",
        regex=True,
        inplace=True,
    )
    ks["insurer"].replace(
        r"VA f\. Eisenb\.u\.Bergbau|Abteilung A",
        "VA f. Eisenb.u.Bergbau Abteilung A",
        regex=True,
        inplace=True,
    )
    # ks["insurer"].replace("Gebietskrankenkassen", "Österr. Gesundheitskasse", inplace=True)

    # Comment next line out to represent the time series break
    ks["insurer"] = ks["insurer"].str.removeprefix("GKK ")
    ks["insurer"] = ks["insurer"].str.strip()
    ks.ffill(inplace=True, limit=1)
    # display(ks)
    ks = pd.wide_to_long(
        ks,
        KS_BASECOLS,
        ["id", "date", "insurer"],
        "employment",
        sep="-",
        suffix="(ArbeiterInnen|Angestellte)",
    ).reset_index()
    mask = np.round(ks["cases"] / ks["insured"] * 1000) != ks["cases_p1000"]
    if mask.any():
        raise ValueError("Mismatch of cases_p1000 at " + ks.loc[mask].to_csv(sep=";"))
    mask = np.round(ks["active_end"] / ks["insured"] * 1000) != ks["active_end_p1000"]
    if mask.any():
        raise ValueError(
            "Mismatch of active_end_p1000 at " + ks.loc[mask].to_csv(sep=";")
        )
    ks.drop(columns=["id", "cases_p1000", "active_end_p1000"], inplace=True)
    for c in KS_BASECOLS:
        if c in ("cases_p1000", "active_end_p1000"):
            continue
        ks[c] = ks[c].astype(int)
    ks.drop_duplicates(
        # Comment the next line to see
        # (a) corrections in 2020 08-10 totals,
        # (b) some changes through ÖGKK -> ÖGK
        subset=["date", "insurer", "employment"],
        keep="last",
        inplace=True,
    )
    ks.set_index(["date", "insurer", "employment"], inplace=True, verify_integrity=True)
    return ks.sort_index()

def main():
    outdir = util.COLLECTROOT / SVDIRNAME
    data = collect_ks()
    outdir.mkdir(parents=True, exist_ok=True)
    data.to_csv(outdir / "ks_all.csv", sep=";", encoding="utf-8")

if __name__ == '__main__':
    main()
