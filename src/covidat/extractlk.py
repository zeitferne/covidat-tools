import argparse
import csv
import html
import itertools
import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import dateparser
from bs4 import BeautifulSoup

tz_vie = ZoneInfo("Europe/Vienna")

TITLE_RE = re.compile(r"<title>([^<]*)</title>")
TS_RE = re.compile(r"\(Stand\s+([^)]+)\s*(?:\)|$)")
TS2_RE = re.compile(r"\(\s*Presseaussendung\s+vom\s+([^)\n<]+)\s*(?:\)<)")
# CARE_HOME_RE = re.compile(
#    r"in (?P<chlocs>\d+|[a-züö]+) oberösterreichischen Alten- und Pflegeheim/?(?:en)?,?"
#    r"(?: (?P<chstaff>\d+|[a-züö/]+).? Mitarbeiter[^\n+.!?]*?(?:und|sowie)?)?"
#    r"(?: (?P<chres>\d+|[a-züö/]+) Bewohner[^\n+.!?]* )?positiv",
#    re.IGNORECASE,
# )
# CARE_HOME_RE_2 = re.compile(
#    r"(?P<chstaff>\d+|[a-züö/]+) Mitarbeiter[^\n+!?]*?(?:und|sowie)? ?"
#    r"(?:.*(?:und |sowie )(?P<chres>\d+|[a-züö/]+)? Bewohne[^\n+!?]* )?von (?P<chlocs>\d+|[a-züö]+?)?"
#    r" ?Alten- und Pflegeheimen in Oberösterreich von Covid-19 betroffen",
#    re.IGNORECASE,
# )

CARE_HOME_RE_LN = re.compile(
    r"[^\n]+?(\d+|[a-züö]+?)?\s+"
    r"(?:Alten- und Pflegeheim(?:en)?\b[^\n]*\bvon Covid[- ]19 betroffen"
    r"|oberösterreichischen Alten- und Pflegeheim[^\n]* positiv)"
    r"[^\n]+",
    re.IGNORECASE,
)

CHSTAFF_RE = re.compile(r"(\d+|[a-züö/]+).? Mitarbeiter", re.IGNORECASE)
CHRES_RE = re.compile(r"(\d+|[a-züö/]+).? Bewohner", re.IGNORECASE)

# print(CARE_HOME_RE_LN.pattern)
CARE_HOME_NONE_STR = "keine Fälle in den oberösterreichischen Alten- und Pflegeheim"


def _hosp_re(marker):
    return re.compile(
        rf"(?P<n>\d+) Covid-19-Patient[^\n]+ auf {marker} sind "
        r"[^\n]*?(?:(?P<n2>[0-9,]+)\s*Personen)?[^\n]*?(?:(?P<r>[0-9,]+)\s*(?:%|Prozent)\s*\)?)?"
        r"\s*nicht vollständig immunisiert",
        re.IGNORECASE,
    )


NST_VACC_RE = _hosp_re("den Normalpflegestationen")
ICU_VACC_RE = _hosp_re("Intensivstationen")
HWS_RE = re.compile(r"[ \t\u00A0]+")

HOSP_WEEKLY_RE = re.compile(
    r"""Kalenderwoche (?P<cw>\d+) wurden von (?:den )?Oberösterreichischen Krankenanstalten
(?:ins)?gesamt (?P<total>\d+) Patientinnen bzw. Patienten mit einer SARS[- ]CoV[- ]2[- ]Infektion stationär
auf Normal- und Intensivstationen neu aufgenommen.[^\n]*
(?P<rnst>[0-9,.]+) Prozent der Patientinnen (?:und|bzw\.) Patienten auf Normalstationen sind nicht
vollständig (?:grund)?immunisiert\.? \(?(?:das|dies) (?:entspricht|sind) (?P<nstunvacc>\d+) Person(?:en)?\)?\.?
Von den in der vergangenen Kalenderwoche aufgenommenen und intensivmedizinisch
zu versorgenden Patientinnen (?:und|bzw\.) Patienten sind (?P<ricu>[0-9,.]+ Prozent|alle)
(?P<invicu>nicht )?vollständig
(?:grund)?immunisiert\.? \(?(?:das|dies) (?:entspricht|sind) (?P<icuunvacc>\d+) Person(?:en)?\)?""".replace(
        "\n", " "
    ),
    re.IGNORECASE,
)

print(HOSP_WEEKLY_RE.pattern)


def processfile(fname: str, ofile: csv.DictWriter, deathfile):
    with open(fname, encoding="utf-8") as f:
        text = f.read()
    processtext(text, fname, ofile, deathfile)


def toint(s):
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        if s.startswith("ein") and "und" not in s:
            return 1
        return {
            "keine": 0,
            "zwei": 2,
            "drei": 3,
            "vier": 4,
            "fünf": 5,
            "sechs": 6,
            "sieben": 7,
            "acht": 8,
            "neun": 9,
            "zehn": 10,
            "elf": 11,
            "zwölf": 12,
        }[s]


def gethospoccupancy(mhosp: re.Match):
    if not mhosp:
        return (None, None)

    n = mhosp.group("n")
    pct = (mhosp.group("r") or "").replace(",", ".")
    n2 = mhosp.group("n2")
    # print(n, n2, pct)
    n2 = n2 if n2 is not None else (int(n) * float(pct) / 100)
    return (n, n2)


def extracthospvax(rtext: str, fname: str):
    rvals = [None] * 4
    mhosp = NST_VACC_RE.search(rtext)
    rvals[:2] = gethospoccupancy(mhosp)
    mhosp = ICU_VACC_RE.search(rtext)
    rvals[2:] = gethospoccupancy(mhosp)
    result = {
        "FZHosp": rvals[0],
        "hospunvax": rvals[1],
        "FZICU": rvals[2],
        "icuunvax": rvals[3],
    }

    mhosp = HOSP_WEEKLY_RE.search(rtext)
    if mhosp:
        # XXX: This will break down if we get zero unvaccinated
        # in either ICU or NST. Probably a new text will be used then.

        def parserate(rate: str):
            suffix = " Prozent"
            if rate.endswith(suffix):
                return float(rate[: -len(suffix)].replace(",", ".")) / 100
            elif rate == "alle":
                return 1
            raise ValueError("Unknown rate: " + rate)

        total = int(mhosp.group("total"))
        float(mhosp.group("rnst").replace(",", ".")) / 100
        ricu = parserate(mhosp.group("ricu"))
        int(mhosp.group("nstunvacc"))
        icuunvacc = int(mhosp.group("icuunvacc"))

        # Looks like the normal station percentage is about all persons, not
        # only newly admitted. Don't use it.

        icutotal = icuunvacc / ricu

        if not mhosp.group("invicu"):
            ricu = 1 - ricu
            icuunvacc = icutotal - icuunvacc

        result.update(
            {
                "FZHospnew": total - icutotal,
                "hospunvaxnew": None,  # Can't calculate
                "FZICUnew": icutotal,
                "icuunvaxnew": icuunvacc,
            }
        )

    return result


UHR_RE = re.compile(r"(\s+\d+) Uhr")


def parsets(tsstr: str, fname: str):
    ts = dateparser.parse(UHR_RE.sub(r"\1:00 Uhr", html.unescape(tsstr)), languages=["de"])
    if not ts:
        raise ValueError(f"Bad date for {fname}: {html.unescape(tsstr)}")
    return ts


def findts(text: str, fname: Path):
    m = TITLE_RE.search(text)
    # breakpoint()
    if not m:
        raise ValueError(f"No title for {fname}")
    mts = TS_RE.search(m.group(1))
    if not mts:
        return None
    ttl_ts = parsets(mts.group(1), fname)
    mts2 = TS2_RE.search(text)
    if mts2:
        ts2 = parsets(mts2.group(1), fname)
        if ts2.date() != ttl_ts.date():
            if fname.name in (
                "234967.htm",
                "250699.htm",
                "250702.htm",
                "272379.htm",
                "272388.htm",
                "276861.htm",
                "288599.htm",
                "289032.htm",
                "289036.htm",
                "289295.htm",
                "290441.htm",
                "290869.htm",
            ):
                pass  # Use date from title
            elif fname.name == "270813.htm":
                return ts2  # Use date from "Presseaussendung vom"
            else:
                raise ValueError(f"Timestamp mismatch: {ttl_ts} vs {ts2} in {fname}")
    return ttl_ts


def processtext(text: str, fname: Path, ofile: csv.DictWriter, deathfile):
    ts = findts(text, fname)
    if ts is None:
        # print("Skipping", fname.name, m.group(1), "no timestamp")
        pass
    else:
        # print(
        #    fname.name,
        #    html.unescape(mts.group(1)),
        #    ts,
        # )
        rtext = HWS_RE.sub(" ", BeautifulSoup(text, "html.parser").get_text()).replace("\r", "\n")
        row = {"Datum": ts, "id": fname.name}
        # with open(fname.stem + ".txt", "w", encoding="utf-8", newline="") as oof:
        #    oof.write(rtext)
        if CARE_HOME_NONE_STR in rtext:
            chvals = (0, 0, 0)
        else:
            # mch = CARE_HOME_RE.search(rtext) or CARE_HOME_RE_2.search(rtext)
            mch = CARE_HOME_RE_LN.search(rtext)
            if mch:
                chvals = [None] * 3
                chvals[0] = mch.group(1)
                if chvals[0] == "von":
                    chvals[0] = None
                m2 = CHSTAFF_RE.search(mch.group(0))
                chvals[1] = m2.group(1) if m2 else None
                if chvals[1] == "und":
                    chvals[1] = None
                m2 = CHRES_RE.search(mch.group(0))
                chvals[2] = m2.group(1) if m2 else None
                if chvals[2] == "und":
                    chvals[2] = None
                try:
                    chvals = [toint(chval) for chval in chvals]
                except KeyError as k:
                    raise ValueError(f"Failed parsing int '{k.args[0]}' in {fname}: {mch.group(0)}") from None
                if all(v is None for v in chvals):
                    raise ValueError(f"No result in {fname}: {mch.group(0)}")
                chvals[1] = chvals[1] or 0
                chvals[2] = chvals[2] or 0
                if chvals[0] is None and sum(chvals[1:]) == 1:
                    chvals[0] = 1
                if not any(chvals):
                    chvals[0] = 0
                if chvals[0] == 0 and (chvals[1] or chvals[2]):
                    raise ValueError(f"Impossible result (a) {chvals} in {fname}: {mch.group(0)}")
                if (
                    chvals[0] is not None
                    and chvals[0] > (chvals[1] or 0) + (chvals[2] or 0)
                    and fname.name not in ("235127.htm", "235261.htm", "284917.htm", "284992.htm")
                ):
                    raise ValueError(f"Impossible result (b) {chvals} in {fname}: {mch.group(0)}")
            else:
                chvals = (None,) * 3
        row.update(
            {
                "chlocs": chvals[0],
                "chstaff": chvals[1],
                "chres": chvals[2],
            }
        )
        row.update(extracthospvax(rtext, fname.name))
        extractdeaths(ts, fname.name, rtext, deathfile)
        ofile.writerow(row)


# 91-jährige Patientin, wohnhaft im Bezirk Gmunden, Vorerkrankungen unbekannt, Todesdatum: 30. Oktober
#   (Salzkammergut Klinikum Bad Ischl-Gmunden-Vöcklabruck, Standort Gmunden)
COND_INNER_PAT = (
    r"(?:(?:mit|ohne|keine)(?: schweren?)? Vorerkrankung(?:en)?|Vorerkrankung(?:en)? unbekannt|Vorerkrankung(?:en)?"
    r" (?:(?:zum Zeitpunkt der Meldung )?noch )?nicht bekannt)")
COND_PAT = rf"(?: (?P<cond>{COND_INNER_PAT})\b\.?,?\s*)"
COND_INNER_RE = re.compile(COND_INNER_PAT, re.IGNORECASE)
BEZ_PREFIX_PAT = r"(?:wohnhaft (?:in |im Bezirk,? )|aus dem Bezirk |aus (?:der Stadt )?|Bezirk )"
BEZ_PREFIX_RE = re.compile(BEZ_PREFIX_PAT, re.IGNORECASE)
AGE_PAT = r"(?P<age>\d+)[-. ]+(?:jähr?i?g?e?r?e?|jähirger|jährigeer)"
DEAD_RES = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        rf"{AGE_PAT} (?P<label>[A-Za-z]+)[,.]? {BEZ_PREFIX_PAT}?(?P<district>[^\n,]+?),?"
        rf"{COND_PAT}?\s*(?:Todesdatum\b(?:[.,:] ?| )"
        rf"(?P<deathdate>[^\n(,]+))?\s*[(,](?!.*Todesdatum)\s*(?P<deathloc>[^\n)]+)",
        rf"{AGE_PAT} (?P<label>[A-Za-z]+)[,.]?"
            rf" {BEZ_PREFIX_PAT}?(?P<district>[^\n,]+?),?"
            rf"{COND_PAT}?\s*"
            r"(?:Todesdatum\b(?:[.,:] ?| )(?P<deathdate>[^\n(,]+))(?:\n|$)(?P<deathloc>never){0}",
        rf"1 (?P<label>[A-Za-z]+){COND_PAT}? \((?P<age>\d+)\), (?P<deathloc>[^\n]+)\s*(?:\n|$)(?P<district>nope)?",
        rf"1 (?P<label>[A-Za-z]+)\s*[,(]\s*(?P<age>\d+)s*[,)]\s*"
            rf"{BEZ_PREFIX_PAT}(?P<district>[^\n,]+),{COND_PAT}?(?P<deathloc>[^\n]+)\s*(?:\n|$)",
        rf"Todesfall im (?P<deathloc>[^\n,]+), {BEZ_PREFIX_PAT}(?P<district>[^\n,]+?),"
            rf" {AGE_PAT} (?P<label>[A-Za-z]+){COND_PAT}?",
        rf"{AGE_PAT} (?P<label>[A-Za-z]+){COND_PAT} {BEZ_PREFIX_PAT}?(?P<district>[^\n,]+?)"
            rf"(?: im (?P<deathloc>[^\n]+))?(?:\n|$)",
        rf"{AGE_PAT} (?P<label>[A-Za-z]+) {BEZ_PREFIX_PAT}?(?P<district>[^\n,]+?){COND_PAT}"
            rf"(?: im (?P<deathloc>[^\n]+))?(?:\n|$)",
        rf"1 (?P<label>[A-Za-z]+) {BEZ_PREFIX_PAT}(?P<district>[^\n,]+)\s*[,(]\s*(?P<age>\d+)s*[,)]{COND_PAT}?"
            rf"\s*(?P<deathloc>[^\n]+)\s*(?:\n|$)",
    ]
)

PAR_TEXT_RE = re.compile(r"\(([^)\n]+)\)")

DEL_RE = re.compile(
    r"\n\s*Nachtrag zur Todesfallmeldung.+\n|"
    r"\n\s*Von den heute am Dashboard des Landes OÖ \([^)]+\) ausgewiesenen Todesfällen,? war.+bereits.+\n|"
    r"wegen möglicher Vorerkrankungen zur Risikogruppe",
    re.IGNORECASE,
)
# print(DEAD_RE_4.pattern)

# print(DEAD_RES[-1].pattern)
DEATH_START_RE = re.compile("Aktuelle Tode|Todesfälle im Zusammenhang mit C", re.IGNORECASE)


def extractdeaths(ts: date, name: str, rtext: str, deathfile):
    if name == "245261.htm":
        rtext = rtext.replace("26.11. Salzkammergut", "26.11. (Salzkammergut")
    elif name == "245594.htm":
        rtext = rtext.replace("01.12. Salzkammergut", "01.12. (Salzkammergut")
    elif name == "245437.htm":
        rtext = rtext.replace("unbekannt, 15.11.", "unbekannt, Todesdatum: 15.11.")
    elif name == "246388.htm":
        rtext = rtext.replace("86-Patient", "86-jähriger Patient")
    elif name == "248172.htm":
        rtext = rtext.replace("21.01. Ordensklinikum", "21.01. (Ordensklinikum")
    elif name == "252256.htm":
        return
    elif name == "252498.htm":
        rtext = rtext.replace("Todesdatum: 30.30.", "Todesdatum: 30.03.")
    elif name == "254359.htm":
        rtext = rtext.replace("Todesdatum: 2.05.201", "Todesdatum: 2.05.2021")
    elif name == "292206.htm":
        rtext = rtext.replace("; Sterbeort:", ", ")

    rtext = DEL_RE.sub("\n", rtext)
    n = 0
    # print(ts, name)
    items = []
    for m in itertools.chain.from_iterable(r.finditer(rtext) for r in DEAD_RES):
        n += 1
        # print(name, n, m)
        hasdate = "deathdate" in m.groupdict() and bool(m.group("deathdate"))
        dt = (
            None
            if not hasdate
            else dateparser.parse(
                m.group("deathdate").strip(),
                languages=["de"],
                date_formats=[
                    "%d.%m.%Y",
                    "%d.%m.%y",
                    "%d.%m.",
                    "%d. %B",
                    "%d.%B",
                    "%d.%m",
                    "%d.0%m.",
                    "%d:%m.",
                ],
                settings={
                    #'DATE_ORDER': 'DMY',
                    "RELATIVE_BASE": ts,
                    "REQUIRE_PARTS": ["day", "month"],
                    "PREFER_DATES_FROM": "past",
                    "PARSERS": ["custom-formats"],
                    "DEFAULT_LANGUAGES": ["de"],
                },
            )
        )
        if dt:
            if dt.year > ts.year:
                dt = datetime(ts.year, dt.month, dt.day, tzinfo=tz_vie)
            if dt > ts and dt.year == ts.year:
                dt = datetime(ts.year - 1, dt.month, dt.day, tzinfo=tz_vie)
            if dt > ts:
                raise ValueError(f"Future-dated death: {dt} in {m[0]} / {name} {ts}")
        if name == "243890.htm" and "wird noch bekannt gegeben" in m.group(0):
            # * 95-jähriger Patient wohnhaft im Bezirk Urfahr-Umgebung mit Vorerkrankungen
            #   (Klinikum wird noch bekannt gegeben).
            # Nachtrag zur Todesfallmeldung vom 3. November 2020: der 95-jährige Mann wohnhaft im
            #   Bezirk Urfahr-Umgebung verstarb mit Vorerkrankungen im BHS Gramastetten.
            items.append(
                [
                    ts,
                    name,
                    dt.date() if dt else dt,
                    m["district"],
                    "BHS Gramastetten",
                    m["age"],
                    "Mann",
                    m["cond"],
                ]
            )
            continue
        vals = list(m.group("district", "deathloc", "age", "label", "cond"))
        if vals[0]:
            condm = COND_INNER_RE.match(vals[0])
            if condm:
                # print(condm)
                vals[-1] = condm.group(0)
                vals[0] = BEZ_PREFIX_RE.sub("", vals[0][: condm.start(0)] + vals[0][condm.end(0) :]).strip()
            parm = PAR_TEXT_RE.search(vals[0])
            if parm:
                vals[1] = parm.group(1)
                vals[0] = BEZ_PREFIX_RE.sub("", vals[0][: parm.start(0)] + vals[0][parm.end(0) :]).strip()
        items.append([ts, name, dt.date() if dt else dt, *vals])
        if hasdate and not dt and m.group("deathdate").strip() != "unbekannt":
            raise ValueError(f"Bad date: '{m.group('deathdate')}' in {name}")
        if not hasdate and not m.group("cond"):
            if (
                "verstorben" not in m[0]
                and "Obduktion" not in m[0]
                and (m.start(0) - DEATH_START_RE.search(rtext).end(0) > 1000)
            ):
                raise ValueError(f"Really a death? '{m.group(0)}' in {name}")
            else:
                # print(name, n, m, m.groupdict())
                n -= 1  # Not included in our heuristic
    expect = max(rtext.lower().count("todesda"), rtext.lower().count("vorerkrank"))
    if expect != n:
        raise ValueError(f"Missing entries in {name}: expected {expect}, found {n}")
    for item in items:
        deathfile.writerow(item)


def runextraction(args):
    done = set()
    try:
        oldfile_h = open("extractlk.csv", encoding="utf-8", newline="\n")
    except FileNotFoundError:
        pass
    else:
        oldfile = csv.DictReader(oldfile_h, delimiter=";")
        for row in oldfile:
            # print(row)
            done.add(row["id"])
    openmode = "w" if not done else "a"
    # sys.exit(f"{openmode=} {len(done)=}")
    # raise
    with (
        open("extractlk.csv", openmode, encoding="utf-8", newline="\n") as ofile_h,
        open("extractlk.dead.csv", openmode, encoding="utf-8", newline="\n") as deathfile_h,
    ):
        ofile = csv.DictWriter(
            ofile_h,
            [
                "Datum",
                "id",
                "chlocs",
                "chstaff",
                "chres",
                "FZHosp",
                "hospunvax",
                "FZICU",
                "icuunvax",
                "FZHospnew",
                "hospunvaxnew",
                "FZICUnew",
                "icuunvaxnew",
            ],
            delimiter=";",
        )
        if not done:
            ofile.writeheader()

        deathfile = csv.writer(deathfile_h, delimiter=";")
        if not done:
            deathfile.writerow(
                [
                    "pubdate",
                    "id",
                    "deathdate",
                    "district",
                    "deathloc",
                    "age",
                    "label",
                    "cond",
                ]
            )
        for fname in Path(args.lkdir).glob("*.htm"):
            if fname.name in done:
                continue
            processfile(fname, ofile, deathfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lkdir")
    args = parser.parse_args()
    runextraction(args)


if __name__ == "__main__":
    main()
