#!/usr/bin/env python3
# coding: utf-8
from os.path import basename
from os.path import expanduser
from sys import argv
from sys import exit
from sys import stderr

import bibtexparser
from bibtexparser.bparser import BibTexParser

#
# Script to remove unwanted keys from bibtex files exported by Mendeley.
#
# Forked from: https://gist.github.com/chaosct/254d896a262d7438dff7
#
# Authors: Carles F. Julià <carles.fernandez(AT)upf.edu>
#          Chris Cummins <chrisc.101(AT)gmail.com>
#
# Requirements: bibtexparser
#   $ pip install bibtexparser
#
# Usage: cleanbib <input-bib>
#


class EmptyBibtexError(Exception):
  pass


# Unwanted keys.
ignored_keys = ("file", "annote", "abstract")

# Output file header.
header = (
  "File generated automatically by 'cleanbib' script.\n\n"
  "**********************************\n"
  "* DO NOT EDIT THIS FILE BY HAND! *\n"
  "**********************************\n\n"
)


# Accepts a string path to a bibfile and returns a bibtexparser
# object. Any error will cause the program to exit.
def readbibfile(path):
  try:
    with open(expanduser(path)) as input_file:
      bib_database = bibtexparser.load(input_file, parser=BibTexParser())
      if not len(bib_database.entries):
        raise EmptyBibtexError
      return bib_database
  # Catch error states.
  except FileNotFoundError:
    print("fatal: file '{file}' not found.".format(file=path), file=stderr)
    exit(1)
  except EmptyBibtexError:
    print(
      "warning: file '{file}' contains no BibTeX entries".format(file=path),
      file=stderr,
    )
    print("To be on the safe side, I'm going to exit now.", file=stderr)
    exit(0)


# Accepts a file and writes a string to it with UTF-8 encoding.
def writeutf8(file, string):
  file.write(string.encode("utf8"))


if __name__ == "__main__":

  if len(argv) != 2:
    print("Usage: {self} <input-bib>".format(self=basename(argv[0])))
    exit(1)

  path = argv[1]

  # Read file in.
  bib_database = readbibfile(path)

  # Filter BibTex database.
  for entry in bib_database.entries:
    for k in ignored_keys:
      entry.pop(k, None)

  # Dump BibTeX to string.
  bibtex_string = bibtexparser.dumps(bib_database)

  # Write file out.
  with open(expanduser(path), "wb") as output_file:
    writeutf8(output_file, header)
    writeutf8(output_file, bibtex_string)

  print("wrote {len} entries.".format(len=len(bib_database.entries)))
