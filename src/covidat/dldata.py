#!/usr/bin/env python3

import argparse
import csv
import dataclasses
import io
import json
import logging
import lzma
import mimetypes
import os
import re
import tomllib
import typing
import urllib.response
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from itertools import chain
from os.path import basename
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile
from zoneinfo import ZoneInfo

from .dlutil import dl_with_header_cache, get_moddate, write_hdr_file
from .util import DATAROOT, DL_TSTAMP_FMT, LOG_FORMAT, parse_statat_date

logger = logging.getLogger(__name__)


TFn = TypeVar("TFn", bound=Callable)


def _register_fn(fn: TFn, registry: dict[str, TFn], prefix: str = ""):
    name = fn.__name__
    if prefix:
        name = name.removeprefix(prefix)
    if name in registry:
        raise ValueError(f"The name {name} is already registered")
    registry[name] = fn
    return fn


DateExtractor = Callable[[urllib.response.addinfourl, bytes], datetime | None]
_date_extractors: dict[str, DateExtractor] = {}


@dataclass(frozen=True, kw_only=True, match_args=False)
class DlCfg:
    dry_run: bool
    dldir: Path
    archive: bool = True
    archive_headers: bool = False
    compress: bool = False
    default_file_extension: str | None = None
    fname_format: str = "{}"
    sortdir_fmt: str = ""
    extract_date: DateExtractor | None = None


def date_extractor(fn: DateExtractor):
    return _register_fn(fn, _date_extractors)


Downloader = Callable[[DlCfg], None]
DownloaderFactory = Callable[[str, dict[str, Any]], Downloader]
_dl_factories: dict[str, DownloaderFactory] = {}


def downloader_factory(fn: DownloaderFactory):
    return _register_fn(fn, _dl_factories, "download_")


def cmp_openfiles(
    f1: io.BufferedIOBase, f2: io.BufferedIOBase, buf1: bytearray | memoryview, buf2: bytearray | memoryview
) -> bool:
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


def cmp_files(
    p1: os.PathLike | str, p2: os.PathLike | str, buf1: bytearray | memoryview, buf2: bytearray | memoryview
) -> bool:
    """Checks if files under paths p1, p2 are equal. buf1 and buf2 must be
    objects of same lenght supporting the buffer protocol, preferably
    memoryviews."""
    assert len(buf1) > 0
    assert len(buf1) == len(buf2)
    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        return cmp_openfiles(f1, f2, buf1, buf2)


def splitbasestem(fname: os.PathLike | str) -> tuple[str, str]:
    fname = basename(fname)
    if not fname.lstrip("."):
        raise ValueError("Invalid file name for stemming: " + fname)
    if not fname:
        return (fname, "")
    # If the name starts with dot, we don't consider it a file separator
    stem, sep, ext = fname[1:].partition(".")
    return fname[0] + stem, sep + ext


def dl_url(url: str, cfg: DlCfg, *, autocommit=True) -> Exception | tuple[bytes, Callable[[], None]] | None:
    fname = PurePosixPath(urlparse(url).path).name
    fstem, fext = splitbasestem(fname)
    fext = fext or cfg.default_file_extension
    fstem = cfg.fname_format.format(fstem)
    if not is_safe_fname(fstem) and not is_safe_fname(fname):
        raise ValueError("Unsafe filename rejected in: " + url)
    hdrfilepath = cfg.dldir / (fstem + "_lasthdr.txt")
    dlts = datetime.now(UTC)
    resp = None
    newpath = None
    try:
        ok, resp, oldheaders = dl_with_header_cache(url, hdrfilepath, dry_run=cfg.dry_run)
        olddate_raw = get_moddate(oldheaders) if oldheaders else None
        olddate = olddate_raw.strftime("%Y-%m-%d %H:%M") if olddate_raw else None
        if not ok:
            status = getattr(resp, "status", "")
            if status == HTTPStatus.NOT_MODIFIED and oldheaders:
                logger.info("H 304 %s (Kept: %s)", url, olddate or oldheaders.get("Etag"))
                return None
            # TODO: Might be unreachable / if-stmt might be mostly useless
            logger.warning("E %s %s %r", status, url, resp)
            return typing.cast(Exception, resp)
        resp = typing.cast(urllib.response.addinfourl, resp)

        data = resp.read()

        def fpath_from_resp(resp) -> tuple[Path, str]:
            hdrs = resp.headers
            ts = None
            if cfg.extract_date is not None:
                ts = cfg.extract_date(resp, data)
            if ts is None:
                ts = get_moddate(hdrs) or dlts
            dt_stamp = ts.strftime(DL_TSTAMP_FMT)
            fext_real = fext or mimetypes.guess_extension(hdrs.get_content_type()) or ""
            if cfg.compress:
                fext_real += ".xz"
            newdir = cfg.dldir
            if cfg.sortdir_fmt:
                newdir /= ts.strftime(cfg.sortdir_fmt)
            return newdir / f"{fstem}_{dt_stamp}{fext_real}", fext_real

        def commit_headers():
            write_hdr_file(resp.headers, hdrfilepath, allow_existing=True)

        def is_header_updated(name):
            oldval = None if oldheaders is None else oldheaders.get(name)
            return resp.headers.get(name) != oldval

        def maybe_commit_headers():
            if is_header_updated("Etag") or is_header_updated("Last-Modified"):
                commit_headers()
            else:
                logger.debug("Headers contain no relevant change, not storing them.")

        def simpname(fpath: Path):
            return splitbasestem(fpath.name)[0].removeprefix(fstem).strip("_")

        newpath, fext = fpath_from_resp(resp) if cfg.archive else (cfg.dldir / fname, fext)
        if cfg.archive and newpath.exists():
            logger.info("Same modification date: %s (Kept: %s)", url, simpname(newpath))
            maybe_commit_headers()
            return None

        newdir = newpath.parent
        newdir.mkdir(parents=True, exist_ok=True)

        dlpath = newpath.with_suffix(newpath.suffix + ".tmp")
        with lzma.open(dlpath, "wb") if cfg.compress else open(dlpath, "wb") as of:
            of.write(data)

        linkpath = cfg.dldir / f"{fstem}_latest{fext}"
        if cfg.archive:
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
                maybe_commit_headers()
                return None
            if newpath.exists():
                raise FileExistsError("Target exists but has different content:" + str(newpath))
            dlpath.rename(newpath)
        else:
            dlpath.replace(newpath)
        if cfg.archive_headers:
            write_hdr_file(resp.headers, newdir / (newpath.stem + "_hdr.txt"), allow_existing=False)
        relpath = newpath.relative_to(DATAROOT)
        logger.info(
            "Dowloaded %s %s",
            url,
            newpath if len(str(newpath)) <= len(str(relpath)) else relpath,
        )
        if cfg.archive:
            linkdst = newpath.relative_to(linkpath.parent)
            linkpath.unlink(missing_ok=True)
            linkpath.symlink_to(linkdst)
        if autocommit:
            commit_headers()

            def commit_headers_fault():
                raise NotImplementedError("Cannot commit after autocommit")

            commit_headers = commit_headers_fault
    except Exception as exc:
        logger.exception("Error downloading %s to %s", url, newpath or cfg.dldir)
        return exc
    finally:
        if resp is not None:
            resp.close()
    return data, commit_headers


@date_extractor
def ages_versiondate(resp: urllib.response.addinfourl, data: bytes) -> datetime | None:
    try:
        with io.BytesIO(data) as data_io, ZipFile(data_io) as zf, zf.open("Version.csv") as verfile, io.TextIOWrapper(
            verfile, encoding="utf-8"
        ) as verfile_s:
            ver = next(iter(csv.DictReader(verfile_s, delimiter=";")))["CreationDate"]
    except Exception as exc:
        md = get_moddate(resp.headers)
        logger.error("Failed extract version, using Last-Modified %s date - 2h", md, exc_info=exc)
        return md - timedelta(hours=2) if md is not None else None
    return datetime.strptime(ver, "%d.%m.%Y %H:%M:%S").replace(tzinfo=ZoneInfo("Europe/Vienna"))


@date_extractor
def medshort_updatedate(_resp: urllib.response.addinfourl, data: bytes) -> datetime | None:
    # Daten zuletzt aktualisiert am: 2022-12-10 00:31:12
    tmatch = re.search(rb"aktualisiert am: ([0-9-]+ [0-9:]+)", data)
    if not tmatch:
        return None
    return datetime.strptime(tmatch.group(1).decode("utf-8"), "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=ZoneInfo("Europe/Vienna")
    )


# For OGD definition see https://go.gv.at/ogdframede
# https://neu.ref.wien.gv.at/at.gv.wien.ref-live/documents/20189/84932/Metadaten_data.gv.at_2.6_DE_20210907.pdf/34e12148-98dd-4e20-8b18-d9f8b4956b31
# (note that Statistik Austria claims to use v2.3, see e.g.
# <https://www.data.gv.at/wp-content/uploads/2013/08/OGD-Metadaten_2-3_2014_11_10_EN.pdf>)


def extract_ogd_moddate(_resp: urllib.response.addinfourl, data: bytes) -> datetime | None:
    meta = json.loads(data)
    # Date-format is string-sortable. Unclear if metadata_modified would be enough,
    # but better safe than sorry
    mod_str = max(
        chain(
            (meta["extras"]["metadata_modified"],),
            (r["last_modified"] for r in meta["resources"] if "last_modified" in r),
        )
    )

    # Timezone is actually unknown, might also be Europe/Vienna
    return parse_statat_date(mod_str)


def dl_at_ogd_set(meta: dict[str, Any], dlcfg: DlCfg) -> None:
    resource: dict[str, Any]
    for resource in meta["resources"]:
        mdate_s = resource.get("last_modified")
        res_dlcfg = dlcfg
        if mdate_s:
            mdate = parse_statat_date(mdate_s)

            def extract_date(_headers, _body, mdate=mdate):
                return mdate

            res_dlcfg = dataclasses.replace(res_dlcfg, extract_date=extract_date)
        dl_url(resource["url"], res_dlcfg)


_always_safe_rng = "-A-Za-z0-9_öÖäÄüÜß+()=!,;"  # Never safe: ?/\
is_safe_fname = re.compile(fr"^[{_always_safe_rng}.]*[{_always_safe_rng}]$").fullmatch


@downloader_factory
def download_statat(_kind: str, src_cfg: dict[str, Any]) -> Downloader:
    urls = typing.cast(list[str], src_cfg.pop("urls"))
    if not urls:
        raise ValueError("Missing source urls for statat downloader")

    def do_download_stat(dlcfg: DlCfg) -> None:
        for url in urls:
            dset_q = parse_qs(urlparse(url).query).get("dataset")
            dset = dset_q[0] if dset_q and len(dset_q) == 1 else None
            if not dset or not is_safe_fname(dset):
                dset = None
            res = dl_url(
                url,
                dataclasses.replace(dlcfg, extract_date=extract_ogd_moddate, fname_format=dset or "{}"),
                autocommit=False,
            )
            if not isinstance(res, tuple):
                continue
            meta_bytes, commit = res
            dl_at_ogd_set(json.loads(meta_bytes), dlcfg)
            commit()

    return do_download_stat


def parse_source_cfg(dlsource: dict[str, Any]) -> Downloader:
    kind = dlsource.pop("kind")
    create_downloader = _dl_factories[kind]
    return create_downloader(kind, dlsource)


def execute_dlset(
    dlset: dict[str, Any],
    rootcfg: dict[str, Any],
    *,
    only_archivable: bool,
    only_enabled: bool,
    tag_map: dict[str, bool],
    dry_run: bool,
):
    dlset = dlset.copy()
    preset = dlset.pop("preset", None)
    if preset is not None:
        dlset = rootcfg["presets"][preset] | dlset
    urls = dlset.pop("urls", None)
    source_cfg = dlset.pop("source", None)
    if not urls and source_cfg is None:
        raise ValueError("dlset without URLs and source encountered")
    if urls is not None and source_cfg is not None:
        raise ValueError("dlset with both urls and source encountered")

    if source_cfg is not None:
        if not isinstance(source_cfg, dict):
            raise ValueError("dlset source must be a table")
        downloader = parse_source_cfg(source_cfg)
    else:

        def downloader(dlcfg: DlCfg):
            for url in urls:
                dl_url(url, dlcfg)

    params: dict[str, Any] = {"dry_run": dry_run, "dldir": DATAROOT / dlset.pop("dir")}

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

    enable = dlset.pop("enable", not only_enabled)
    tags = dlset.pop("tags", ()) or ("UNTAGGED",)

    if dlset:
        raise ValueError("Unknown keys in dlset: " + ", ".join(map(repr, dlset.keys())))

    dlcfg = DlCfg(**params)

    for tag in tags:
        override = tag_map.get(tag)
        if override is not None:
            enable = override

    if not enable or only_archivable and not params.get("archive", True):
        return

    downloader(dlcfg)


def execute_config(
    cfg: dict[str, Any],
    *,
    only_archivable: bool,
    only_enabled: bool,
    tag_map: dict[str, bool],
    dry_run: bool,
):
    for dlset in cfg["dlsets"]:
        execute_dlset(
            dlset, cfg, only_archivable=only_archivable, tag_map=tag_map, dry_run=dry_run, only_enabled=only_enabled
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile")
    parser.add_argument("--loglevel")
    parser.add_argument("-c", "--config-path", required=True, type=Path)
    parser.add_argument("--only-archivable", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-tags", nargs="+", action="extend")
    parser.add_argument(
        "--disable-others", action="store_true", help="Disable dlsets not explicitly enabled using --enable-tags"
    )

    args = parser.parse_args()

    logging.basicConfig(
        filename=args.logfile,
        level=args.loglevel.upper() if args.loglevel else "INFO",
        format=LOG_FORMAT,
    )

    with open(args.config_path, "rb") as cfgfile:
        cfg = tomllib.load(cfgfile)
    tag_map = {t: True for t in args.enable_tags} if args.enable_tags else {}
    execute_config(
        cfg,
        only_archivable=args.only_archivable,
        tag_map=tag_map,
        dry_run=args.dry_run,
        only_enabled=args.disable_others,
    )


if __name__ == "__main__":
    main()
