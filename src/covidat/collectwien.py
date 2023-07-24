from datetime import datetime
from pathlib import Path
import csv
from . import util
import html
from itertools import chain
import re

HWS_RE = re.compile(r"[ \t\u00A0]+")
INT_PAT = r"(?:\d+|(?:[0-9]{1,3}(?:\.[0-9]{3})+))"
HOTLINE_RE = re.compile(rf"\b1450 hat ({INT_PAT}) Anruf")
TSTAMP_RE = re.compile(r'<meta property="article:published_time" content="([^"]+)" ?/>')
HOSP_ALL_RE = re.compile(
    rf"Derzeit sind ({INT_PAT}) Personen wegen oder mit COVID-19 in Wien in Spitalsbeh|andlung",
    re.IGNORECASE,
)


def _hosp_re(station: str, cat: str) -> re.Pattern:
    return re.compile(
        rf"COVID-19-Patient\*?innen in {station}pflege auf einer {cat}COVID-Station\*?: ?({INT_PAT})",
        re.IGNORECASE,
    )


HOSP_N_C = _hosp_re("Normal", "")
HOSP_N_NC = _hosp_re("Normal", "Nicht-")
HOSP_N_PC = _hosp_re("Normal", "Post-")

HOSP_I_C = _hosp_re("Intensiv", "")
HOSP_I_NC = _hosp_re("Intensiv", "Nicht-")
HOSP_I_PC = _hosp_re("Intensiv", "Post-")


def match_number(pat: re.Pattern, text: str) -> int:
    match = pat.search(text)
    if not match:
        raise KeyError(f"Could not find a match for: {pat.pattern}")
    return int(match.group(1).replace(".", ""))


def collectwien(dirname, outname):
    frames = []
    firstrow = True
    with open(outname, "w", encoding="utf-8", newline="\n") as outfile:
        writer = csv.DictWriter(
            outfile,
            (
                "Datum",
                "HospCov",
                "HospOutCov",
                "HospPostCov",
                "ICUCov",
                "ICUOutCov",
                "ICUPostCov",
                "HotlineCalls",
            ),
            delimiter=";",
        )
        writer.writeheader()
        dirp = Path(dirname)
        for fname in sorted(chain(dirp.glob("*.html"), dirp.glob("*.htm"))):
            with open(fname, encoding="utf-8") as f:
                fdata = HWS_RE.sub(" ", html.unescape(f.read()))
            mdate_match = TSTAMP_RE.search(fdata)
            if not mdate_match:
                raise ValueError(f"No timestamp in {fname}")
            mdate = datetime.fromisoformat(mdate_match.group(1)).date().isoformat()
            try:
                if mdate not in ("2023-01-03", "2023-01-10", "2022-10-22"):
                    hospdata = {
                        "HospCov": match_number(HOSP_N_C, fdata),
                        "HospOutCov": match_number(HOSP_N_NC, fdata),
                        "HospPostCov": match_number(HOSP_N_PC, fdata),
                        "ICUCov": match_number(HOSP_I_C, fdata),
                        "ICUOutCov": match_number(HOSP_I_NC, fdata),
                        "ICUPostCov": match_number(HOSP_I_PC, fdata),
                    }
                else:
                    hospdata = {
                        "HospCov": None,
                        "HospOutCov": None,
                        "HospPostCov": None,
                        "ICUCov": None,
                        "ICUOutCov": None,
                        "ICUPostCov": None,
                    }
                writer.writerow(
                    {
                        "Datum": mdate,
                        "HotlineCalls": (
                            match_number(HOTLINE_RE, fdata)
                            if mdate < "2023-04-28"
                            and mdate not in ("2022-11-02", "2022-11-26", "2023-02-23")
                            else None
                        ),
                        **hospdata,
                    }
                )
            except KeyError as e:
                raise ValueError(f"Error in {fname}: {e}") from e


def main():
    collectwien(util.COLLECTROOT / "wien", util.COLLECTROOT / "wien/wien.csv")


if __name__ == "__main__":
    main()
