#!/usr/bin/env python3.6
"""
Rebuild the stderrs using the new from_str() constructor method.
"""
import dsmith
import progressbar
import sqlalchemy as sql
from dsmith.sol.db import *


# from dsmith.opencl.db import *


if __name__ == "__main__":
  import logging

  logging.getLogger().setLevel(logging.INFO)
  dsmith.langs.mklang("sol")

  with Session() as s:
    print("Replacing stderrs ...")
    bar = progressbar.ProgressBar(
      max_value=s.query(Result).count(), redirect_stdout=True
    )

    try:
      q = s.query(Result).options(sql.orm.joinedload(Result.stderr))

      for i, result in bar(enumerate(q)):

        # get the prior stderr
        stderr, truncated = result.stderr.stderr, result.stderr.truncated

        # construct a new stderr
        new_stderr = Stderr.from_str(s, stderr)
        new_stderr.truncated = truncated

        # update the result's stderr
        result.stderr = new_stderr

        if not i % 10000:
          s.commit()
    finally:
      s.commit()
