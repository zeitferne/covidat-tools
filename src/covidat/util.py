import os
from pathlib import Path
from zoneinfo import ZoneInfo

Openable = str | bytes | os.PathLike

DL_TSTAMP_FMT = "%Y%m%d_%H%M%S"

DATAROOT = Path(os.getenv("COVAT_DATA_ROOT", "../covidat-data/data"))
COLLECTROOT = Path(os.getenv("COVAT_COLLECT_ROOT", "tmpdata"))
TZ_AT = ZoneInfo("Europe/Vienna")
LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
