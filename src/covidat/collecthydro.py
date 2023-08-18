#!/usr/bin/env python3

import csv
import json
import logging
import lzma
from datetime import UTC, datetime
from itertools import zip_longest

from . import util

logger = logging.getLogger(__name__)


def collecthydro(prefixname: str, *, has_uww: bool, insubdir: bool, namecol: str = "name") -> None:
    subdir = "covid/abwassermonitoring"
    outname = util.COLLECTROOT / subdir / (prefixname + "_all.csv.xz")
    outname.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(outname, "wt", encoding="utf-8", newline="", preset=1) as outfile:
        writer = csv.writer(outfile, delimiter=";")
        writer.writerow(["FileDate", "Datum", namecol, *(("uwwcode",) if has_uww else ()), "y"])
        n = 0
        for fname in sorted(
            (util.DATAROOT / subdir).glob("????/" + (f"/{prefixname}/" if insubdir else "") + prefixname + "_*_*.json")
        ):
            # print(fname)
            with open(fname, "rb") as f:
                try:
                    fdata = json.load(f)
                except json.JSONDecodeError as e:
                    logger.warning("Failed parsing JSON in %s: %s", fname, e)
                    continue
            fdate = (
                datetime.strptime(fname.stem.removeprefix(prefixname + "_"), util.DL_TSTAMP_FMT)
                .replace(tzinfo=UTC)
                .isoformat()
            )
            for linedata in fdata["data"]:
                name = linedata[namecol]
                uww = (linedata["uwwcode"],) if has_uww else ()
                for x, y in zip_longest(linedata["x"], linedata["y"]):
                    writer.writerow([fdate, x, name, *uww, y])
            n += 1
        logger.info("%s: Collected %d files", prefixname, n)


def main() -> None:
    logging.basicConfig(
        level="INFO",
        format=util.LOG_FORMAT,
    )

    collecthydro("natmon_02", has_uww=False, insubdir=False, namecol="typ")
    collecthydro("blverlauf", has_uww=True, insubdir=True)
    collecthydro("natmon_01", has_uww=True, insubdir=False)


if __name__ == "__main__":
    main()
