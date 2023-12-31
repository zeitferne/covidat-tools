import os
import os.path
import re
import shlex
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree, which
from subprocess import check_call


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--sync-to-s3")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--notebook", type=Path)
    args = parser.parse_args()

    pdir = Path(__file__).absolute().parent
    while pdir != pdir.root:
        pdir = pdir.parent
        if (pdir / ".git").is_dir():
            break
    else:
        raise ValueError("Repo root not found")

    ppat = re.escape(str(pdir.parent).replace("\\", "/"))
    if len(ppat) > 2 and ppat[1] == ":":  # noqa: PLR2004
        ppat = f"(?:{ppat})|(?:{ppat[2:]})"
    ppat = f"(?:{ppat})|(?:{re.escape(os.path.realpath(Path.home()))})"
    ppat = f"(?:{ppat})|(?:{re.escape(str(Path.home()))})"
    ppat = ppat.replace("/", r"[/\\]")
    pdir_pat = re.compile(ppat.encode("utf-8"), re.IGNORECASE)

    output_dir = Path("gh-pages/export")
    if args.notebook is None and output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)

    nbfiles = (args.notebook,) if args.notebook else Path(".").glob("*.ipynb")

    for nbfile in nbfiles:
        no_input = b"# @export: --no-input" in nbfile.read_bytes()[:16_000]

        exe = which("jupyter")
        if not exe:
            raise ValueError("Could not find jupyter executable")
        exportargs = [
            exe,
            "nbconvert",
            "--to=html",
            "--ExtractOutputPreprocessor.enabled=true",
            "--HTMLExporter.mathjax_url=",
            "--HTMLExporter.require_js_url=",
            f"--output-dir={output_dir}",
            str(nbfile),
        ]
        if no_input:
            exportargs.append("--no-input")
        if args.execute:
            exportargs.append("--execute")
        filedir = output_dir / (nbfile.stem + "_files")
        if filedir.exists():
            rmtree(filedir)
        print("Running", shlex.join(exportargs), flush=True, file=sys.stderr)
        check_call(exportargs)
        fname = output_dir / nbfile.with_suffix(".html")
        with open(fname, "rb") as htmlfile:
            html = htmlfile.read()
        html = html.replace(
            b"<title>",
            b"""
<style>
        html, body { max-width: 1000px; margin: auto; }
        .jp-OutputArea-child { overflow: auto !important; }
</style>
<title>""",
            1,
        )
        html = pdir_pat.sub(b"...", html)
        with open(fname, "wb") as htmlfile:
            htmlfile.write(html)

    if args.sync_to_s3:
        subprocess.check_call(["aws", "s3", "sync", output_dir, "s3://" + args.sync_to_s3, "--delete"])


if __name__ == "__main__":
    main()
