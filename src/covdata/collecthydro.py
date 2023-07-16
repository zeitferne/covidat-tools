#!/usr/bin/env python3

import json
from datetime import datetime
from pathlib import Path
import csv
import lzma
from itertools import zip_longest
import util

def collecthydro(prefixname, insubdir=False):
    subdir = "covid/abwassermonitoring"
    outname = util.COLLECTROOT / subdir / (prefixname + "_all.csv.xz")
    with lzma.open(outname, "wt", encoding="utf-8", newline="\n", preset=1) as outfile:
        writer = csv.writer(outfile, delimiter=";")
        writer.writerow(["FileDate", "Datum", "name", "uwwcode", "y"])
        for fname in sorted((util.DATAROOT / subdir).glob(
                "????/" + (f"/{prefixname}/" if insubdir else "")
                + prefixname + "_*_*.json")):
            #print(fname)
            with open(fname, "rb") as f:
                fdata = json.load(f)
            fdate = datetime.strptime(
                fname.stem.removeprefix(prefixname + "_"), util.DL_TSTAMP_FMT
            ).isoformat()
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
