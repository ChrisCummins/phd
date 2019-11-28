#!/usr/bin/env python3.6
"""
Add stderrs linecount and charcount metadata, and truncate original strings.
"""
import dsmith
import progressbar
from dsmith.opencl.db import *

# from dsmith.sol.db import *


if __name__ == "__main__":
  # dsmith.langs.mklang("solidity")
  dsmith.langs.mklang("opencl")

  with Session() as s:
    print("Setting metadata for stderrs ...")
    bar = progressbar.ProgressBar(
      max_value=s.query(Stderr).count(), redirect_stdout=True
    )

    fs.mkdir("stderrs")
    for i, stderr in enumerate(s.query(Stderr).yield_per(2000)):
      bar.update(i)

      buf = result.toProtobuf().SerializeToString()
      checksum = crypto.sha1(buf)

      with open(f"stderrs/{checksum}.pb", "wb") as f:
        f.write(buf)

      stderr.linecount = len(stderr.stderr.split("\n"))
      stderr.charcount = len(stderr.stderr)
      stderr.truncated = stderr.charcount > stderr.max_chars
      stderr.stderr = stderr.stderr[: stderr.max_chars]

      # if not i % 2000:
      #     s.commit()
    s.commit()
