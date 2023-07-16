#!/usr/bin/env python3

from http.client import HTTPSConnection, HTTPResponse
from datetime import date, timedelta
from pathlib import Path
from shutil import copyfileobj
import sys
from urllib.error import HTTPError
import dlutil
import util

from typing import Optional

# NB: Need to manually add:
# https://presse.wien.gv.at/presse/2022/10/10/aktualisiert-corona-virus-aktuelle-kennzahlen-der-stadt-wien

URL_PATH_FMTS = (
    "/presse/{}/corona-aktuelle-kennzahlen-aus-wien",
    "/presse/{}/corona-virus-aktuelle-kennzahlen-aus-wien",
    "/presse/{}/corona-virus-aktuelle-kennzahlen-der-stadt-wien",
)

URL_DT_FMT = "%Y/%m/%d"


def requestkennzahlen_single(conn: HTTPSConnection, dt: date, lastfmt_idx: int) -> int:
    dirp = util.COLLECTROOT / "wien/kennzahlen" / dt.strftime("%Y")
    fpath = dirp / ("kennz_wien_" + dt.strftime(util.DL_TSTAMP_FMT) + ".htm")
    if fpath.exists():
        return lastfmt_idx

    fmt_idx = lastfmt_idx
    for try_idx in range(len(URL_PATH_FMTS)):
        fmt_idx = (lastfmt_idx + try_idx) % len(URL_PATH_FMTS)
        urlfmt = URL_PATH_FMTS[fmt_idx]
        url = urlfmt.format(dt.strftime(URL_DT_FMT))
        # print(f"{dt=} {url=}")
        conn.request("GET", url, headers={"From": dlutil.FROM_EMAIL})
        with conn.getresponse() as resp:
            is_notfound = False
            if resp.status == 200:
                data = resp.read()
                is_notfound = (
                    b"Internet-Adresse (URL) ist auf unserem Server nicht oder nicht mehr vorhanden"
                    in data
                )
                if not is_notfound:
                    dirp.mkdir(exist_ok=True, parents=True)
                    with fpath.open("xb") as ofile:
                        ofile.write(data)
                    return fmt_idx
            resp.read()
            if is_notfound or resp.status == 404:
                continue
            raise HTTPError(url, resp.status, resp.reason, resp.headers, None)
    print("No data found for", dt, dt.weekday(), file=sys.stderr)
    return lastfmt_idx


def requestkennzahlen(conn: HTTPSConnection) -> None:
    begdate = date(2020, 3, 5)
    edate = date(2023, 7, 1)
    dt = begdate
    lastfmt_idx = 0
    while dt < edate:
        # print(f"{dt=}")
        lastfmt_idx = requestkennzahlen_single(conn, dt, lastfmt_idx)
        dt += timedelta(1)


def main() -> None:
    conn = HTTPSConnection("presse.wien.gv.at")
    try:
        requestkennzahlen(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
