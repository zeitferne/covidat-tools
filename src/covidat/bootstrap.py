#!/usr/bin/env python3

"""Helper for bootstrapping the bot that reported daily deaths.

Note that the bot (cmpdead) relied on being called from dldata.py, but that functionality
was removed (since the AGES data was also shut down).
"""

import os
import os.path
import shlex
import shutil
import sys
from subprocess import check_call


def is_venv():
    return hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)


def main():
    if not is_venv():
        check_call([sys.executable, *shlex.split("-m venv .venv")])
        env = os.environ.copy()
        bindir = os.path.abspath(".venv/Scripts" if os.name == "nt" else ".venv/bin")
        env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
        if os.name == "nt":
            check_call([".venv/Scripts/python.exe", __file__], env=env)
            return
        else:
            venvpy = ".venv/bin/python"
            os.execle(venvpy, venvpy, __file__, env)
    check_call([sys.executable, *shlex.split("-m pip install -U pip")])
    pipbin = shutil.which("pip")
    if not pipbin:
        raise FileNotFoundError("Cannot find pip")
    check_call([pipbin, *shlex.split("install -U setuptools wheel")])
    check_call([pipbin, *shlex.split("install -U -r"), os.path.join(os.path.dirname(__file__), "requirements.txt")])


if __name__ == "__main__":
    main()
