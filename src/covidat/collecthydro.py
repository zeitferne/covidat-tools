#!/usr/bin/env python3

import csv
import json
import lzma
from datetime import UTC, datetime
from itertools import zip_longest

from . import util


def collecthydro(prefixname: str, insubdir: bool = False):
    subdir = "covid/abwassermonitoring"
    outname = util.COLLECTROOT / subdir / (prefixname + "_all.csv.xz")
    outname.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(outname, "wt", encoding="utf-8", newline="\n", preset=1) as outfile:
        writer = csv.writer(outfile, delimiter=";")
        writer.writerow(["FileDate", "Datum", "name", "uwwcode", "y"])
        for fname in sorted(
            (util.DATAROOT / subdir).glob("????/" + (f"/{prefixname}/" if insubdir else "") + prefixname + "_*_*.json")
        ):
            # print(fname)
            with open(fname, "rb") as f:
                fdata = json.load(f)
            fdate = (
                datetime.strptime(fname.stem.removeprefix(prefixname + "_"), util.DL_TSTAMP_FMT)
                .replace(tzinfo=UTC)
                .isoformat()
            )
            for linedata in fdata["data"]:
                name = linedata["name"]
                uww = linedata["uwwcode"]
                for x, y in zip_longest(linedata["x"], linedata["y"]):
                    writer.writerow([fdate, x, name, uww, y])


def main():
    collecthydro("blverlauf", insubdir=True)
    collecthydro("natmon_01")


if __name__ == "__main__":
    main()
