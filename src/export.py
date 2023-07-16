from subprocess import check_call
from shutil import which, rmtree
import os
import os.path
import subprocess
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--sync-to-s3")
    args = parser.parse_args()

    output_dir = "export"
    if os.path.exists(output_dir):
        rmtree(output_dir)
    os.mkdir(output_dir)
    exportargs =  [
        which("jupyter"),
        "nbconvert",
        "--to=html",
        "--no-input",
        "--ExtractOutputPreprocessor.enabled=true",
        "--output-dir=" + output_dir,
        "covidat.ipynb",
    ]
    if not args.only_export:
        exportargs.append("--execute")
    check_call(exportargs)
    fname = os.path.join(output_dir, "covidat.html")
    with open(fname, "rb") as htmlfile:
        html = htmlfile.read()
    html = html.replace(
        b"<title>",
        b"""
<meta name="robots" content="noindex">
<style>
        html, body { max-width: 800px; margin: auto; }
</style>
<title>""",
        1,
    )
    with open(os.path.join(output_dir, "index.html"), "wb") as htmlfile:
        htmlfile.write(html)
    os.unlink(fname)

    if args.sync_to_s3:
        subprocess.check_call(["aws", "s3", "sync", output_dir, "s3://" + args.sync_to_s3, "--delete"])


if __name__ == "__main__":
    main()
