from pathlib import Path
import os
import typing

Openable = typing.Union[str, bytes, os.PathLike]

DL_TSTAMP_FMT = "%Y%m%d_%H%M%S"

DATAROOT = Path(os.getenv("COVAT_DATA_ROOT", "coronaDAT_patch"))
COLLECTROOT = Path(os.getenv("COVAT_COLLECT_ROOT", "coronaDAT_patch"))
