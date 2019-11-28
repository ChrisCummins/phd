#!/usr/bin/env python3.6
"""
Add stderrs linecount and charcount metadata, and truncate original strings.
"""
import dsmith
import progressbar
from dsmith.opencl.db import *

# from dsmith.sol.db import *


if __name__ == "__main__":
  import logging

  logging.getLogger().setLevel(logging.DEBUG)
  dsmith.langs.mklang("opencl")

  with Session() as s:
    print("Setting metadata for stderrs ...")
    bar = progressbar.ProgressBar(
      max_value=s.query(Stderr).count(), redirect_stdout=True
    )

    try:
      for i, stderr in enumerate(s.query(Stderr).yield_per(2000)):
        bar.update(i)

        stderr.linecount = len(stderr.stderr.split("\n"))
        stderr.charcount = len(stderr.stderr)
        stderr.truncated = stderr.charcount > stderr.max_chars
        stderr.stderr = stderr.stderr.rstrip()[: stderr.max_chars]
    finally:
      s.commit()
