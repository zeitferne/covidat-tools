#!/usr/bin/env python3

import argparse
import csv
import io
import locale
import logging
import lzma
import mimetypes
import re
import sys
import tomllib
import typing
import urllib.response
from collections.abc import Callable
from datetime import datetime, timedelta
from os.path import basename
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Optional, Tuple
from urllib.parse import urlparse
from zipfile import ZipFile

from .dlutil import dl_with_header_cache, get_moddate, write_hdr_file
from .util import DATAROOT, DL_TSTAMP_FMT

logger = logging.getLogger(__name__)
DateExtractor = Callable[[urllib.response.addinfourl, bytes], Optional[datetime]]
_date_extractors: dict[str, DateExtractor] = {}


def date_extractor(fn: DateExtractor):
    if fn.__name__ in _date_extractors:
        raise ValueError(f"The name {fn.__name__} is already registered")
    _date_extractors[fn.__name__] = fn
    return fn


def cmp_openfiles(f1, f2, buf1, buf2) -> bool:
    """Checks if files objects f1, f2 are equal. buf1 and buf2 must be
    objects of same lenght supporting the buffer protocol, preferably
    memoryviews."""
    while True:
        n1, n2 = f1.readinto(buf1), f2.readinto(buf2)
        if n1 != n2:
            # assert n1 == 0 or n2 == 0
            return False
        if n1 == 0:  # EOF
            return True
        if buf1[:n1] != buf2[:n2]:
            return False


def cmp_files(p1, p2, buf1, buf2) -> bool:
    """Checks if files under paths p1, p2 are equal. buf1 and buf2 must be
    objects of same lenght supporting the buffer protocol, preferably
    memoryviews."""
    assert len(buf1) > 0
    assert len(buf1) == len(buf2)
    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        return cmp_openfiles(f1, f2, buf1, buf2)


def splitbasestem(fname: str | PurePath) -> tuple[str, str]:
    fname = basename(fname)
    if not fname.lstrip("."):
        raise ValueError("Invalid file name for stemming: " + fname)
    if not fname:
        return (fname, "")
    # If the name starts with dot, we don't consider it a file separator
    stem, sep, ext = fname[1:].partition(".")
    return fname[0] + stem, sep + ext


def dl_url(
    url: str,
    dldir: Path,
    *,
    archive=True,
    archive_headers=False,
    compress=False,
    default_file_extension=None,
    # str.format()ed with the file basename. Components may be added before
    # or after it. Should not contain extensions and must not contain path
    # separators
    fname_format: str = "{}",
    sortdir_fmt: str = "",
    extract_date: Optional[DateExtractor] = None,
    dry_run: bool,
) -> Optional[Exception]:
    fname = PurePosixPath(urlparse(url).path).name
    fstem, fext = splitbasestem(fname)
    fext = fext or default_file_extension
    fstem = fname_format.format(fstem)
    hdrfilepath = dldir / (fstem + "_lasthdr.txt")
    dlts = datetime.utcnow()
    resp = None
    newpath = None
    try:
        ok, resp, oldheaders = dl_with_header_cache(url, hdrfilepath, dry_run=dry_run)
        olddate_raw = get_moddate(oldheaders) if oldheaders else None
        if olddate_raw:
            olddate = olddate_raw.strftime("%Y-%m-%d %H:%M")
        else:
            olddate = None
        if not ok:
            status = getattr(resp, "status", "")
            if status == 304 and oldheaders:
                logger.info("H 304 %s (Kept: %s)", url, olddate or oldheaders.get("Etag"))
            else:
                logger.warning("E %s %s %r", status, url, resp)
            return None
        resp = typing.cast(urllib.response.addinfourl, resp)

        data = resp.read()

        def fpath_from_resp(resp) -> Tuple[Path, str]:
            hdrs = resp.headers
            ts = None
            if extract_date is not None:
                ts = extract_date(resp, data)
            if ts is None:
                ts = get_moddate(hdrs) or dlts
            dt_stamp = ts.strftime(DL_TSTAMP_FMT)
            fext_real = fext or mimetypes.guess_extension(hdrs.get_content_type()) or ""
            if compress:
                fext_real += ".xz"
            newdir = dldir
            if sortdir_fmt:
                newdir /= ts.strftime(sortdir_fmt)
            return newdir / f"{fstem}_{dt_stamp}{fext_real}", fext_real

        def commit_headers():
            write_hdr_file(resp.headers, hdrfilepath)

        def simpname(fpath: Path):
            return olddate or splitbasestem(fpath.name)[0].removeprefix(fstem).strip("_")

        newpath, fext = fpath_from_resp(resp) if archive else (dldir / fname, fext)
        if archive and newpath.exists():
            logger.info("Same modification date: %s (Kept: %s)", simpname(newpath))
            commit_headers()
            return None

        newdir = newpath.parent
        newdir.mkdir(parents=True, exist_ok=True)

        dlpath = newpath.with_suffix(newpath.suffix + ".tmp")
        of = lzma.open(dlpath, "wb") if compress else open(dlpath, "wb")
        with of:
            of.write(data)

        linkpath = dldir / f"{fstem}_latest{fext}"
        if archive:
            if linkpath.exists() and cmp_files(
                linkpath,
                dlpath,
                bytearray(io.DEFAULT_BUFFER_SIZE),
                bytearray(io.DEFAULT_BUFFER_SIZE),
            ):
                logger.info(
                    "Same file content: %s (Kept: %s)",
                    url,
                    simpname(linkpath.readlink()),
                )
                dlpath.unlink()
                commit_headers()  # Update headers (might contain etag/last-modified)
                return None
            if newpath.exists():
                raise FileExistsError("Target exists but has different content:" + str(newpath))
            dlpath.rename(newpath)
        else:
            dlpath.replace(newpath)
        if archive_headers:
            write_hdr_file(resp.headers, newdir / (newpath.stem + "_hdr.txt"), allow_existing=False)
        relpath = newpath.relative_to(DATAROOT)
        logger.info(
            "Dowloaded %s %s",
            url,
            newpath if len(str(newpath)) <= len(str(relpath)) else relpath,
        )
        if archive:
            linkdst = newpath.relative_to(linkpath.parent)
            linkpath.unlink(missing_ok=True)
            linkpath.symlink_to(linkdst)
        commit_headers()
    except Exception as exc:
        logger.exception("Error downloading %s to %s", url, newpath or dldir)
        return exc
    finally:
        if resp is not None:
            resp.close()
    return None


@date_extractor
def ages_versiondate(resp: urllib.response.addinfourl, data: bytes) -> Optional[datetime]:
    try:
        with io.BytesIO(data) as data_io, ZipFile(data_io) as zf, zf.open("Version.csv") as verfile, io.TextIOWrapper(
            verfile, encoding="utf-8"
        ) as verfile_s:
            ver = next(iter(csv.DictReader(verfile_s, delimiter=";")))["CreationDate"]
    except Exception as exc:
        md = get_moddate(resp.headers)
        logger.error("Failed extract version, using Last-Modified %s date - 2h", md, exc_info=exc)
        return md - timedelta(hours=2) if md is not None else None
    return datetime.strptime(ver, "%d.%m.%Y %H:%M:%S")


@date_extractor
def medshort_updatedate(resp: urllib.response.addinfourl, data: bytes) -> Optional[datetime]:
    # Daten zuletzt aktualisiert am: 2022-12-10 00:31:12
    tmatch = re.search(rb"aktualisiert am: ([0-9-]+ [0-9:]+)", data)
    if not tmatch:
        return None
    return datetime.strptime(tmatch.group(1).decode("utf-8"), "%Y-%m-%d %H:%M:%S")


def execute_dlset(
    dlset: dict[str, Any],
    cfg: dict[str, Any],
    only_archivable: bool,
    tag_map: dict[str, bool],
    dry_run: bool,
):
    dlset = dlset.copy()
    preset = dlset.pop("preset", None)
    if preset is not None:
        dlset = cfg["presets"][preset] | dlset
    urls = dlset.pop("urls", None)
    if not urls:
        raise ValueError("dlset without URLs encountered")
    dldir = DATAROOT / dlset.pop("dir")
    params: dict[str, Any] = {"dry_run": dry_run}

    for k in (
        "archive",
        "archive_headers",
        "sortdir_fmt",
        "compress",
        "default_file_extension",
        "fname_format",
    ):
        v = dlset.pop(k, None)
        if v is not None:
            params[k] = v

    extract_date = dlset.pop("extract_date", None)
    if extract_date is not None:
        params["extract_date"] = _date_extractors[extract_date]

    enable = dlset.pop("enable", True)
    tags = dlset.pop("tags", ()) or ("UNTAGGED",)

    if dlset:
        raise ValueError("Unknown keys in dlset: " + ", ".join(map(repr, dlset.keys())))

    for tag in tags:
        override = tag_map.get(tag)
        if override is not None:
            enable = override

    if not enable or only_archivable and not params.get("archive", True):
        return

    for url in urls:
        dl_url(url, dldir, **params)


def execute_config(
    cfg: dict[str, Any],
    *,
    only_archivable: bool,
    tag_map: dict[str, bool],
    dry_run: bool,
):
    for dlset in cfg["dlsets"]:
        execute_dlset(
            dlset,
            cfg,
            only_archivable=only_archivable,
            tag_map=tag_map,
            dry_run=dry_run,
        )


def main() -> None:
    locale.setlocale(locale.LC_ALL, "de_AT.UTF-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile")
    parser.add_argument("--loglevel")
    parser.add_argument("-c", "--config-path", required=True, type=Path)
    parser.add_argument("--only-archivable", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-tags", nargs="+", action="extend")

    args = parser.parse_args()

    logging.basicConfig(
        filename=args.logfile,
        level=args.loglevel.upper() if args.loglevel else args.loglevel,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    )

    with open(args.config_path, "rb") as cfgfile:
        cfg = tomllib.load(cfgfile)
    tag_map = {t: True for t in args.enable_tags} if args.enable_tags else {}
    execute_config(cfg, only_archivable=args.only_archivable, tag_map=tag_map, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
