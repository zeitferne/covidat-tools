import ssl


def create_ctx(*args, **kwargs):
    result = ssl._create_unverified_context(*args, **kwargs)
    result.set_ciphers("DEFAULT@SECLEVEL=1")
    return result


ssl._create_default_https_context = create_ctx

from urllib.error import HTTPError
from datetime import datetime, date, timedelta
import sys
from http.client import HTTPSConnection
from urllib.parse import urlsplit
import glob
from . import dlutil
from . import util

CERT_DATA_URL_FMT = (
    "https://info.gesundheitsministerium.gv.at/data/archiv/COVID19_vaccination_municipalities_%Y%m%d.csv"
)
CERT_DATA_AG_URL_FMT = (
    "https://info.gesundheitsministerium.at/data/archiv/COVID19_vaccination_doses_agegroups_%Y%m%d.csv"
)
CERT_FIRST = datetime(2021, 10, 29, 23, 59, 59)


def readold(fname, fromdate):
    try:
        existing = open(
            fname,
            encoding="utf-8",
            newline="\n",
        )
    except FileNotFoundError:
        return fromdate, False
    lastline = None
    with existing:
        for line in existing:
            if ";" in line:
                lastline = line
    if not lastline:
        return fromdate, False
    rawdate = lastline.split(";", 1)[0]
    # We need to start from two days later, as the URL at dt contains
    # the data with timestamp dt - 1 and we want to go to URL dt + 1
    return datetime.fromisoformat(rawdate) + timedelta(2), True


def ahook(name, args):
    if name == "http.client.connect":
        print(name, args, file=sys.stderr, flush=True)


sys.addaudithook(ahook)


def append_dl_file(url_fmt, fname, fromdate, glob_fmt=None):
    dt, hasold = readold(fname, fromdate)
    data = []

    def add_data_file(resp):
        data.append((resp.read().decode("utf-8").replace("\r", "").removesuffix("\n") + "\n").splitlines(True))

    missing = 0
    conn = HTTPSConnection(urlsplit(datetime.today().strftime(url_fmt)).netloc)
    try:
        while True:
            print(dt, flush=True)
            try:
                url = urlsplit(dt.strftime(url_fmt))
                conn.request("GET", url.path + url.query, headers={"From": dlutil.FROM_EMAIL})
                with conn.getresponse() as resp:
                    if resp.status != 200:
                        resp.read()
                        raise HTTPError(url.geturl(), resp.status, resp.reason, resp.headers, None)
                    if resp.headers["Content-Type"] != "text/csv":
                        # If you request a nonexistent page, you don't get a 404
                        # but a redirect to an HTML page.
                        print(dt, "Got", resp.headers["Content-Type"])
                        break
                    add_data_file(resp)
            except HTTPError as e:
                print(dt, e, file=sys.stderr, flush=True)

                if (dt + timedelta(1)).astimezone(None).date() >= date.today() or missing > 10:
                    break
                if glob_fmt:
                    pat = dt.strftime(glob_fmt)
                    files = list(glob.glob(pat, recursive=False))
                    if files:
                        with open(files[-1], "rb") as dfile:
                            print("Opened", dfile, "for", dt)
                    else:
                        print("No files matched for", pat)
                        # add_data_file(dfile)
                missing += 1
            dt += timedelta(1)
    finally:
        conn.close()
    if not data:
        print("No data found for", fname, file=sys.stderr)
        return
    with open(
        fname,
        "a" if hasold else "x",
        encoding="utf-8",
        newline="\n",
    ) as of:
        if not hasold:
            of.writelines(data[0])
        for file in data if hasold else data[1:]:
            of.writelines(file[1:])  # Strip heading.


def main():
    append_dl_file(
        CERT_DATA_URL_FMT,
        util.DATAROOT / "COVID19_vaccination_municipalities_timeline.csv",
        CERT_FIRST,
        glob_fmt="coronaDAT_patch/misc/COVID19_vaccination_municipalities_%Y%m%d_??????.csv",
    )
    append_dl_file(
        CERT_DATA_AG_URL_FMT,
        util.DATAROOT / "COVID19_vaccination_doses_agegroups_v202206_timeline.csv",
        CERT_FIRST,
    )


if __name__ == "__main__":
    main()
