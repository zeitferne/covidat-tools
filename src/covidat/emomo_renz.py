#!/bin/env python3

import sys
from csv import DictReader
from pathlib import Path


def main():
    for fpath in Path(sys.argv[1]).glob("*.csv"):
        print(fpath)
        with open(fpath, newline="") as cf:
            reader = DictReader(cf, delimiter=";")
            if "zscore" not in reader.fieldnames:
                continue
            data = list(reader)
            bycountry = "country" in reader.fieldnames and len(frozenset(row["country"] for row in data)) > 1
        grps = frozenset(row["group"] for row in data)
        lastweek = max(row["week"] for row in data)
        minweek = min(row["week"] for row in data)
        countrytag = "-by-country" if bycountry else ""
        agtag = "-by-agegroup" if len(grps) > 1 else "-" + next(iter(grps))
        fname = f"zscores{countrytag}{agtag}_w{lastweek}"
        newpath = fpath.with_stem(fname)
        if newpath == fpath:
            continue
        if newpath.exists():
            raise ValueError(f"Duplicate fname: {newpath} (from {fpath}, mw={minweek})")
        fpath.rename(newpath)


if __name__ == "__main__":
    main()
