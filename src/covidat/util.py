import os
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

Openable = str | bytes | os.PathLike

DL_TSTAMP_FMT = "%Y%m%d_%H%M%S"

DATAROOT = Path(os.getenv("COVAT_DATA_ROOT", "../covidat-data/data"))
COLLECTROOT = Path(os.getenv("COVAT_COLLECT_ROOT", "tmpdata"))
TZ_AT = ZoneInfo("Europe/Vienna")
LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"


def parse_statat_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)


def fdate_from_fname(s: Openable) -> datetime:
    p = Path(s)
    return datetime.strptime("_".join(p.stem.rsplit("_", 3)[-2:]), DL_TSTAMP_FMT).replace(tzinfo=UTC)
