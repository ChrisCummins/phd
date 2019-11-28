#!/usr/bin/env python2
"""

Usage:
  $ python run_filters.py <results_dir>
"""
from __future__ import print_function

import os
import pkgutil
import sys

import filters
import google.protobuf.text_format
from proto import deepsmith_pb2


def AllFilterModules():
  """Yields an iterator over all filter modules."""
  for importer, modname, _ in pkgutil.iter_modules(filters.__path__):
    yield importer.find_module(modname).load_module(modname)


def LoadResultProto(path):
  """Load a Result proto from file."""
  with open(path) as f:
    proto = google.protobuf.text_format.Merge(f.read(), deepsmith_pb2.Result())
  return proto


def WriteResultProto(path, result):
  """Write a Result proto to file."""
  with open(path, "wt") as f:
    f.write(google.protobuf.text_format.MessageToString(result))


def main():
  """Main entry point."""
  results_dir = "."
  filter_modules = list(AllFilterModules())
  all_result_paths = [
    os.path.join(results_dir, x) for x in os.listdir(results_dir)
  ]

  for result_path in all_result_paths:
    try:
      result = LoadResultProto(result_path)
    except google.protobuf.text_format.ParseError as e:
      print(
        "Failed to read result: '{result_path}'.\n{e}".format(
          result_path=result_path, e=e
        ),
        file=sys.stderr,
      )
      continue

    for module in filter_modules:
      print(
        "Running filter: '{modname}' on '{basename}' ...".format(
          modname=module.__name__, basename=os.path.basename(result_path)
        )
      )
      result = module.Filter(result)
      if not result:
        break

    if result:
      WriteResultProto(result_path, result)
    else:
      print("Filtered result: '{result_path}'.".format(result_path=result_path))
      os.unlink(result_path)


if __name__ == "__main__":
  main()
