#!/usr/bin/env python
"""Run linters on source files in this repository.

Usage:
  $ ./tools/lint.py [path ...]

By default, only modified files will be linted. If one or more paths are
provided, all git tracked files in those directories will be linted.
"""
from __future__ import print_function

import argparse
import os
import random
import subprocess
import sys
import time


# The path to the root of the PhD repository, i.e. the directory which this file
# is in.
# WARNING: Moving this file may require updating this path!
_PHD_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
assert os.path.isdir(os.path.join(_PHD_ROOT, ".git"))

# Ad-hoc import of //tools/code_style/linters:linters_lib.py
LINTERS_LIB = os.path.join(_PHD_ROOT, "tools/code_style/linters/linters_lib.py")
if sys.version_info >= (3, 5):
  import importlib

  spec = importlib.util.spec_from_file_location("linters_lib", LINTERS_LIB)
  linters_lib = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(linters_lib)
else:
  import imp

  linters_lib = imp.load_source("linters_lib", LINTERS_LIB)


def Ignore(path):
  if path.startswith("deeplearning/clgen/tests/data/"):
    return True
  elif path.startswith("datasets/benchmarks/gpgpu/amd-app-sdk-3.0"):
    return True
  elif path.startswith("datasets/benchmarks/gpgpu/npb-3.3"):
    return True
  elif path.startswith("datasets/benchmarks/gpgpu/nvidia-4.2"):
    return True
  elif path.startswith("datasets/benchmarks/gpgpu/parboil-0.2"):
    return True
  elif path.startswith("datasets/benchmarks/gpgpu/polybench-gpu-1.0"):
    return True
  elif path.startswith("datasets/benchmarks/gpgpu/rodinia-3.1"):
    return True
  elif path.startswith("datasets/benchmarks/gpgpu/shoc-1.1.5"):
    return True
  elif path == "system/dotfiles/usr/share/Dropbox/dropbox.py":
    return True
  elif path == "third_party/opencl/inlined/cl.h":
    return True
  elif path == "util/freefocus/jekyll/js/markdown.min.js":
    return True
  return False


def main(argv):
  os.chdir(_PHD_ROOT)

  parser = argparse.ArgumentParser()
  parser.add_argument("paths", action="store", nargs="*")
  parser.add_argument(
    "--all",
    action="store_true",
    default=False,
    help="Lint all tracked files in the repository.",
  )
  parser.add_argument(
    "--ignore",
    action="store_true",
    default=False,
    help="Ignore a default list of files.",
  )
  parser.add_argument(
    "--verbose",
    action="store_true",
    default=False,
    help="Print verbose messages.",
  )

  args = parser.parse_args(argv)

  if args.all:
    output = subprocess.check_output(
      ["git", "ls-files"], universal_newlines=True
    )
    lint_candidates = output.rstrip().split("\n")
    lint_candidates = [l for l in lint_candidates if os.path.isfile(l)]
  elif args.paths:
    lint_candidates = (
      linters_lib.ExecOrDie(["find"] + args.paths + ["-type", "f"])
      .rstrip()
      .split("\n")
    )
  else:
    lint_candidates = set(
      linters_lib.GetGitDiffFilesOrDie(staged=False)
      + linters_lib.GetGitDiffFilesOrDie(staged=True)
    )
    # Changed files may include deleted files.
    lint_candidates = [f for f in lint_candidates if os.path.isfile(f)]

  if args.ignore:
    lint_candidates = [l for l in lint_candidates if not Ignore(l)]

  # "Load balance" the linter chunks by shuffling the order.
  lint_candidates = list(lint_candidates)
  random.shuffle(lint_candidates)

  if args.verbose:
    print("Linting:")
    for path in sorted(lint_candidates):
      print(" ", path)
    print()

  linters = linters_lib.LinterActions(lint_candidates, verbose=args.verbose)

  start_time = time.time()
  linters_lib.Print(
    "Performing", len(linters.paths_with_actions), "lint actions ...", end=" "
  )
  linters.RunOrDie()
  print("{:.3f}s".format(time.time() - start_time))

  for path in sorted(linters.modified_paths):
    print("   ", path)


if __name__ == "__main__":
  try:
    main(sys.argv[1:])
  except KeyboardInterrupt:
    print("\ninterrupt", file=sys.stderr)
