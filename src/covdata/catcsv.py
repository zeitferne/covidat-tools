import contextlib
from datetime import datetime
import io
from pathlib import Path
from argparse import ArgumentParser, Namespace
from glob import iglob
import shutil
import sys
from itertools import chain

def collect_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("fglob", nargs="+")
    parser.add_argument("-o", "--output")
    return parser.parse_args()

def execute_action(args: Namespace):
    inputs = tuple(chain.from_iterable(sorted(iglob(pat)) for pat in args.fglob))
    if not inputs:
        raise ValueError("No inputs.")
    #raise ValueError(list(inputs))
    binout = sys.stdout.buffer
    output = open(args.output, "wb") if args.output else binout
    isfirst = True
    with output if output is not binout else contextlib.nullcontext():
        for input in inputs:
            with open(input, "rb") as inputfile:
                if not isfirst:
                    inputfile.readline()
                shutil.copyfileobj(inputfile, output)
                inputfile.seek(-1, io.SEEK_CUR)
                if inputfile.read(1) != b'\n':
                    output.write(b"\n")
            isfirst = False
def main() -> None:
  args = collect_args()
  execute_action(args)
    

if __name__ == '__main__':
    main()