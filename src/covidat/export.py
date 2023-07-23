import os
import os.path
import re
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree, which
from subprocess import check_call


def main():
    parser = ArgumentParser()
    parser.add_argument("--sync-to-s3")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    pdir = Path(__file__).absolute().parent.parent
    assert (pdir / ".git").is_dir(), "Expected git dir at " + str(pdir)
    pdir = re.escape(str(pdir.parent).replace("\\", "/"))
    if len(pdir) > 2 and pdir[1] == ":":
        pdir = f"(?:{pdir})|(?:{pdir[2:]})"
    pdir = f"(?:{pdir})|(?:{re.escape(os.path.realpath(Path.home()))})"
    pdir = f"(?:{pdir})|(?:{re.escape(str(Path.home()))})"
    pdir = pdir.replace("/", r"[/\\]")
    pdir_pat = re.compile(pdir.encode("utf-8"), re.IGNORECASE)

    output_dir = "gh-pages/export"
    if os.path.exists(output_dir):
        rmtree(output_dir)
    os.mkdir(output_dir)

    for nbfile in Path(".").glob("*.ipynb"):
        no_input = b"#@export: --no-input" in nbfile.read_bytes()[:16_000]

        exportargs = [
            which("jupyter"),
            "nbconvert",
            "--to=html",
            "--ExtractOutputPreprocessor.enabled=true",
            "--output-dir=" + output_dir,
            str(nbfile),
        ]
        if no_input:
            exportargs.append("--no-input")
        if args.execute:
            exportargs.append("--execute")
        check_call(exportargs)
        fname = output_dir / nbfile.with_suffix(".html")
        with open(fname, "rb") as htmlfile:
            html = htmlfile.read()
        html = html.replace(
            b"<title>",
            b"""
<meta name="robots" content="noindex">
<style>
        html, body { max-width: 1000px; margin: auto; }
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
