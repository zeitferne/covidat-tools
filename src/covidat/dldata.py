#!/usr/bin/env python3

import argparse
import csv
import dataclasses
import html
import inspect
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
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, TypeVar
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen
from zipfile import ZipFile
from zoneinfo import ZoneInfo

from .dlutil import create_request, dl_with_header_cache, get_moddate, write_hdr_file
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


DateExtractor = (
    Callable[[urllib.response.addinfourl, bytes], datetime | None]
    | Callable[[urllib.response.addinfourl], datetime | None]
)
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
    extract_date: DateExtractor | datetime | None = None
    use_disposition_fname: bool = False
    fname_re_sub: tuple[re.Pattern, str] | None = None


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


def get_dl_base_fname(url: str, cfg: DlCfg) -> tuple[str, str]:
    urlparts = urlparse(url)
    fname = PurePosixPath(urlparts.path).name
    fstem, fext = splitbasestem(fname)
    if not fstem or fstem in ("cdscontent", "load"):
        qparts = parse_qs(urlparts.query)
        cid = qparts.get("contentid")
        if cid and len(cid) == 1:
            fstem += cid[0]

    fext = fext or cfg.default_file_extension or ""
    fstem = cfg.fname_format.format(fstem)
    if not is_safe_fname(fstem) and not is_safe_fname(fname):
        raise ValueError("Unsafe filename rejected in: " + url)
    return fstem, fext


def dlpath_from_date(fstem: str, fext: str, ts: datetime | None, cfg: DlCfg) -> Path:
    if cfg.compress:
        fext += ".xz"
    newdir = cfg.dldir
    if ts is None:
        assert not cfg.sortdir_fmt
        dt_stamp = ""
    else:
        dt_stamp = "_" + ts.strftime(DL_TSTAMP_FMT)
        if cfg.sortdir_fmt:
            newdir /= ts.strftime(cfg.sortdir_fmt)
    return newdir / f"{fstem}{dt_stamp}{fext}", fext


def dlpath_from_resp(
    fstem: str, fext: str, data: bytes | None, resp: urllib.response.addinfourl, cfg: DlCfg
) -> tuple[Path, str]:
    hdrs = resp.headers
    hdr_fname = hdrs.get_filename()
    if hdr_fname:
        if cfg.fname_re_sub:
            new_fname = cfg.fname_re_sub[0].sub(cfg.fname_re_sub[1], hdr_fname)
            if new_fname != hdr_fname:
                logger.debug("Replaced %r with %r", hdr_fname, new_fname)
                hdr_fname = new_fname
        safe = is_safe_fname(hdr_fname)
        if not safe:
            if cfg.use_disposition_fname:
                raise ValueError(f"Configuration requests use of unsafe disposition filename: {hdr_fname}")
            hdr_fname = None
    if hdr_fname:
        hdr_stem, hdr_ext = splitbasestem(hdr_fname)
        if cfg.use_disposition_fname:
            fstem, fext = hdr_stem or fstem, hdr_ext or fext
    else:
        hdr_ext = ""
    fext_real = fext or hdr_ext or mimetypes.guess_extension(hdrs.get_content_type()) or ""
    ts = (
        None
        if not cfg.archive
        else get_moddate(hdrs) or datetime.now(UTC)
        if cfg.extract_date is None
        else cfg.extract_date
        if isinstance(cfg.extract_date, datetime)
        else cfg.extract_date(resp)
        if data is None
        else cfg.extract_date(resp, data)
    )
    return dlpath_from_date(fstem, fext_real, ts, cfg)


def dl_url(url: str, cfg: DlCfg, *, autocommit=True) -> Exception | tuple[bytes, Callable[[], None]] | None:
    if cfg.extract_date is not None and not cfg.archive:
        raise ValueError("extract_date is not None despite not archive")
    if cfg.archive_headers and not cfg.archive:
        raise ValueError("archive_headers despite not archive")
    fstem, fext = get_dl_base_fname(url, cfg)
    hdrfilepath = cfg.dldir / (fstem + "_lasthdr.txt")
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

        n_params_with_body = 2
        if (
            callable(cfg.extract_date)
            and sum(
                a.default is inspect.Signature.empty for a in inspect.signature(cfg.extract_date).parameters.values()
            )
            == n_params_with_body
        ):
            data = resp.read()
        else:
            data = None

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

        newpath, fext = dlpath_from_resp(fstem, fext, data, resp, cfg)
        if cfg.archive and newpath.exists():
            logger.info("Same modification date: %s (Kept: %s)", url, simpname(newpath))
            maybe_commit_headers()
            return None

        newdir = newpath.parent
        newdir.mkdir(parents=True, exist_ok=True)

        if data is None:
            data = resp.read()

        dlpath = write_data_tmp(newpath, data, cfg)

        linkpath = cfg.dldir / f"{fstem}_latest{fext}" if cfg.archive else newpath
        if linkpath.exists() and cmp_files(
            linkpath,
            dlpath,
            bytearray(io.DEFAULT_BUFFER_SIZE),
            bytearray(io.DEFAULT_BUFFER_SIZE),
        ):
            logger.info(
                "Same file content: %s (Kept: %s)",
                url,
                simpname(linkpath.readlink()) if cfg.archive else (olddate or "?"),
            )
            dlpath.unlink()
            maybe_commit_headers()
            return None
        if cfg.archive:
            if newpath.exists():
                raise FileExistsError("Target exists but has different content:" + str(newpath))
            dlpath.rename(newpath)
        else:
            dlpath.replace(newpath)
        if cfg.archive_headers:
            write_hdr_file(resp.headers, newdir / (newpath.stem + "_hdr.txt"), allow_existing=False)
        log_download(url, newpath)
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


def log_download(url: str, newpath: PurePath):
    relpath = newpath.relative_to(DATAROOT)
    logger.info("Downloaded %s %s", url, newpath if len(str(newpath)) <= len(str(relpath)) else relpath)


def write_data_tmp(newpath: Path, data: bytes, cfg: DlCfg) -> Path:
    dlpath = newpath.with_suffix(newpath.suffix + ".tmp")
    with lzma.open(dlpath, "wb") if cfg.compress else open(dlpath, "wb") as of:
        of.write(data)
    return dlpath


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


def dl_at_ogd_set(meta: dict[str, Any], dlcfg: DlCfg, *, exclude_url: Callable[[str], bool]) -> None:
    resource: dict[str, Any]
    for resource in meta["resources"]:
        if exclude_url(resource["url"]):
            logger.debug("Excluded via URL filter: %s", resource["url"])
            continue
        mdate_s = resource.get("last_modified")
        res_dlcfg = dlcfg
        if mdate_s:
            mdate = parse_statat_date(mdate_s)
            res_dlcfg = dataclasses.replace(res_dlcfg, extract_date=mdate)
        dl_url(resource["url"], res_dlcfg)


_conservative_save_rng = "A-Za-z0-9"
_always_safe_rng = f"-{_conservative_save_rng}_öÖäÄüÜß+()=!,;"  # Never safe: ?/\
# Careful with dots. We need to allow them (almost every filename contains them),
# but they must not occur more than once in sequence
is_safe_fname = re.compile(fr"^(?:\.?[{_always_safe_rng}])*\.?[{_conservative_save_rng}]$").fullmatch


@downloader_factory
def download_statat(_kind: str, src_cfg: dict[str, Any]) -> Downloader:
    src_cfg = src_cfg.copy()
    urls = typing.cast(list[str], src_cfg.pop("urls"))
    if not urls:
        raise ValueError("Empty source urls for statat downloader")
    exclude_url_regex = src_cfg.pop("exclude_url_regex", None)
    exclude_url = re.compile(exclude_url_regex).search if exclude_url_regex else lambda _url: False
    if src_cfg:
        raise ValueError("Unknown keys in statat config: " + ", ".join(map(repr, src_cfg.keys())))

    def do_download_stat(dlcfg: DlCfg) -> None:
        for url in urls:
            dset_q = parse_qs(urlparse(url).query).get("dataset")
            dset = dset_q[0] if dset_q and len(dset_q) == 1 else None
            if not dset or not is_safe_fname(dset):
                dset = None
            res = dl_url(
                url,
                dataclasses.replace(dlcfg, extract_date=extract_ogd_moddate, fname_format=dset + "_{}" or "{}"),
                autocommit=False,
            )
            if not isinstance(res, tuple):
                continue
            meta_bytes, commit = res
            dl_at_ogd_set(json.loads(meta_bytes), dlcfg, exclude_url=exclude_url)
            commit()

    return do_download_stat


def download_unique_links_re(url: str, source_re: re.Pattern, dlcfg: DlCfg) -> None:
    dlcfg = dataclasses.replace(dlcfg, archive=False, archive_headers=False)
    request = create_request(url)
    resp: urllib.response.addinfourl
    if dlcfg.dry_run:
        data = b""
    else:
        with urlopen(request) as resp:  # noqa: S310
            data = resp.read()
    urllistfilename = dlcfg.dldir / (".".join(p for p in get_dl_base_fname(url, dlcfg) if p) + "_urls.csv")
    csv_cols = ("url", "name")
    urllist: dict[str, str]
    try:
        with open(urllistfilename, newline="", encoding="utf-8") as urllistfile:
            urllist = {r["url"]: r["name"] for r in csv.DictReader(urllistfile)}
    except FileNotFoundError:
        urllist = {}
    newurllist: dict[str, str] = {}
    text = data.decode("utf-8", errors="replace")
    matches = 0
    hits = 0
    try:
        for match in source_re.finditer(text):
            matches += 1
            content_url = html.unescape(match.group(1))
            urlname = urllist.get(content_url) or newurllist.get(content_url)
            if urlname is not None:
                logger.debug("Same URL: %s (%s)", content_url, urlname)
                newurllist[content_url] = urlname
                continue
            hits += 1
            with urlopen(create_request(content_url)) as resp:  # noqa: S310
                data = resp.read()
            fstem, fext = get_dl_base_fname(content_url, dlcfg)
            newpath = dlpath_from_resp(fstem, fext, data, resp, dlcfg)[0]
            dlpath = write_data_tmp(newpath, data, dlcfg)
            if newpath.exists():
                raise FileExistsError(f"URL {content_url} not in list but {newpath} exists already")
            dlpath.rename(newpath)
            newurllist[content_url] = newpath.name
            log_download(content_url, newpath)
    except:
        newurllist = urllist | newurllist
        raise
    finally:
        # TODO: This is still not safe against crashes/poweroffs. We'll be left with
        # inconsistent data that will block us from donwloading anything
        with io.StringIO() if dlcfg.dry_run else open(
            urllistfilename, "w", newline="", encoding="utf-8"
        ) as urllistfile:
            urlwriter = csv.DictWriter(urllistfile, csv_cols)
            urlwriter.writeheader()
            urlwriter.writerows({"url": k, "name": v} for k, v in newurllist.items())
    logger.log(logging.INFO if matches else logging.WARNING, "%s: %d/%d new URLs", url, hits, matches)


@downloader_factory
def scan_unique_links_re(kind: str, src_cfg: dict[str, Any]) -> Downloader:
    # TODO Factor out common config parsing, see download_statat
    src_cfg = src_cfg.copy()
    urls = typing.cast(list[str], src_cfg.pop("urls"))
    if not urls:
        raise ValueError(f"Empty source urls for {kind} downloader")
    source_re = re.compile(src_cfg.pop("regex"))
    if source_re.groups < 1:
        raise ValueError("No capturing groups in source regex:" + source_re.pattern)
    if src_cfg:
        raise ValueError(f"Unknown keys in {kind} config: " + ", ".join(map(repr, src_cfg.keys())))

    def download_unique_links_re_all(dlcfg: DlCfg) -> None:
        for url in urls:
            download_unique_links_re(url, source_re, dlcfg)

    return download_unique_links_re_all


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
        "use_disposition_fname",
    ):
        v = dlset.pop(k, None)
        if v is not None:
            params[k] = v

    extract_date = dlset.pop("extract_date", None)
    if extract_date is not None:
        params["extract_date"] = _date_extractors[extract_date]

    fname_re_sub = dlset.pop("fname_re_sub", None)
    if fname_re_sub is not None:
        re_sub_param_cnt = 2
        if not isinstance(fname_re_sub, list) or len(fname_re_sub) != re_sub_param_cnt:
            raise ValueError("Bad value for fname_re_sub: " + repr(fname_re_sub))
        params["fname_re_sub"] = (re.compile(fname_re_sub[0]), fname_re_sub[1])

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
        try:
            execute_dlset(
                dlset, cfg, only_archivable=only_archivable, tag_map=tag_map, dry_run=dry_run, only_enabled=only_enabled
            )
        except Exception:
            logger.exception(
                "Failed executing dlset (into %s): %s", dlset.get("dir"), dlset.get("urls") or dlset.get("source")
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
