#!/usr/bin/env python3.6
import sys

from progressbar import ProgressBar

from labm8.py import crypto
from labm8.py import fs

if __name__ == "__main__":
  inpath = sys.argv[1]
  outdir = sys.argv[2]
  print(f"reading from {inpath} into {outdir}")

  assert fs.isfile(inpath)
  assert not fs.exists(outdir) or fs.isdir(outdir)
  fs.mkdir(outdir)

  with open(inpath) as infile:
    text = infile.read()

  kernels = text.split("// ==== START SAMPLE ====")
  kernels = [kernel.strip() for kernel in kernels if kernel.strip()]
  print(len(kernels), "kernels")

  sha1s = [crypto.sha1_str(kernel) for kernel in kernels]
  for kernel, sha1 in ProgressBar()(list(zip(kernels, sha1s))):
    with open(f"{outdir}/{sha1}.txt", "w") as outfile:
      print(kernel, file=outfile)
