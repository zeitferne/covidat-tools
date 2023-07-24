import os
import typing
from pathlib import Path

Openable = typing.Union[str, bytes, os.PathLike]

DL_TSTAMP_FMT = "%Y%m%d_%H%M%S"

DATAROOT = Path(os.getenv("COVAT_DATA_ROOT", "../covidat-data/data"))
COLLECTROOT = Path(os.getenv("COVAT_COLLECT_ROOT", "tmpdata"))
