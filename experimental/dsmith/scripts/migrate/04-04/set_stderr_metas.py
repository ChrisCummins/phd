#!/usr/bin/env python3.6
"""
Recompute the compiler assertions, unreachables, and stackdumps for all stderr
records.
"""
import dsmith
import progressbar
from dsmith.opencl.db import *


if __name__ == "__main__":
  dsmith.langs.mklang("opencl")  # Initializes database engine

  with Session() as s:
    print("Setting metadata for stderrs ...")
    bar = progressbar.ProgressBar(
      max_value=s.query(Stderr).count(), redirect_stdout=True
    )

    for i, stderr in enumerate(paginate(s.query(Stderr), page_size=1000)):
      bar.update(i)

      lines = stderr.stderr.split("\n")
      stderr.assertion = Stderr._get_assertion(s, lines)
      stderr.unreachable = Stderr._get_unreachable(s, lines)
      stderr.stackdump = Stderr._get_stackdump(s, lines)

      if not i % 1000:
        s.commit()
    s.commit()
