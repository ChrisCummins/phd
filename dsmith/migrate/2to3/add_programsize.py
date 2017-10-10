#!/usr/bin/env python3
"""
Create test harnesses for CLgen programs using cldrive.
"""
import cldrive
import clgen
import os
import sys
import sqlalchemy as sql
import subprocess
import json
import numpy as np
import datetime

from collections import deque
from itertools import product
from argparse import ArgumentParser
from labm8 import crypto, fs
from pathlib import Path
from progressbar import ProgressBar
from time import time
from typing import List
from tempfile import NamedTemporaryFile

import analyze
import db
import util
import clgen_mkharness
from db import *


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    parser.add_argument("--commit", action="store_true",
                        help="Commit changes (default is dry-run)")
    args = parser.parse_args()

    db.init(args.hostname)

    to_del = deque()

    with Session(commit=True) as s:
        for tables in [CLSMITH_TABLES, CLGEN_TABLES]:
            num_progs = s.query(sql.sql.func.count(tables.programs.id)).scalar()
            if num_progs:
                print(f"Setting {tables.name} program sizes")
                progs = s.query(tables.programs)
                for i, program in enumerate(ProgressBar(max_value=num_progs)(progs)):
                    program.size = len(program.src)
                    if i and not i % 1000:
                        s.commit()
    print("done.")
