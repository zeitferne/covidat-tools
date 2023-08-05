#!/usr/bin/env python3

import csv
import json
import logging
import lzma
from datetime import UTC, datetime
from itertools import zip_longest

from . import util

logger = logging.getLogger(__name__)


def collecthydro(prefixname: str, *, insubdir: bool):
    subdir = "covid/abwassermonitoring"
    outname = util.COLLECTROOT / subdir / (prefixname + "_all.csv.xz")
    outname.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(outname, "wt", encoding="utf-8", newline="\n", preset=1) as outfile:
        writer = csv.writer(outfile, delimiter=";")
        writer.writerow(["FileDate", "Datum", "name", "uwwcode", "y"])
        n = 0
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
            n += 1
        logger.info("%s: Collected %d files", prefixname, n)


def main():
    logging.basicConfig(
        level="INFO",
        format=util.LOG_FORMAT,
    )

    collecthydro("blverlauf", insubdir=True)
    collecthydro("natmon_01", insubdir=False)


if __name__ == "__main__":
    main()
