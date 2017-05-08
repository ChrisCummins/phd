#!/usr/bin/env python3
"""
The Reproducer-er (TM)

Usage: ./the-reproducerer

What it does
------------
  Figure out what OpenCL devices we have on this system
  Fetch all suspicious entries from the DB for the available OpenCL devices
  For each suspicious entry:
    If it is a build failure:
      Build the program again. Reproducible?
"""
import clgen
import os
import sqlalchemy as sql

from argparse import ArgumentParser
from labm8 import fs
from pathlib import Path
from progressbar import ProgressBar

import db
from db import *


def reproduce(testcase):
    if is_build_failure(testcase):
        if reproduce_build_failure(testcase):
            c = serialize_build_failure(testcase)
            if reproduce_c_build_failure(c):
                print("TODO: bug report")
            else:
                print("could not reproduce using C code")
        else:
            print("could not reproduce")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    args = parser.parse_args()

    db.init(args.hostname)
    session = db.make_session()

    print("done.")
