"""Expand shorthand conference and journal names to their full version.

E.g. "CGO" -> "International Symposium on Code Generation and Optimization".

This uses a manually curated set of expansion rules.
"""
import sys

import bibtexparser

from labm8.py import app
from labm8.py import bazelutil
from labm8.py import jsonutil

FLAGS = app.FLAGS

app.DEFINE_input_path(
  "bibtex_path", None, "Path of bibtex file to process in place"
)

_acronyms_json_path = bazelutil.DataPath("phd/docs/acronyms.json")
expansions = jsonutil.read_file(_acronyms_json_path)

_ignored_probable_acronyms = set()


def RewriteName(entry, title, translation_table):
  """Re-write the 'title' key in 'entry' using the translation table."""
  if title not in entry:
    return

  if (
    entry[title].upper() == entry[title]
    and entry[title] not in translation_table
    and entry[title] not in _ignored_probable_acronyms
  ):
    _ignored_probable_acronyms.add(entry[title])
    print(
      f"Ignoring probable acryonym {len(_ignored_probable_acronyms)}: "
      f'"{entry[title]}"',
      file=sys.stderr,
    )
    return

  entry[title] = translation_table.get(entry[title], entry[title])


def ExpandBibtexShorthandInPlace(bibtex, expansions) -> None:
  """Expand shorthand names in-place."""
  for entry in bibtex.entries:
    if entry["ENTRYTYPE"] == "inproceedings":
      RewriteName(entry, "booktitle", expansions["inproceedings"])
    elif entry["ENTRYTYPE"] == "article":
      RewriteName(entry, "journal", expansions["article"])


def BibtexToString(bibtex) -> str:
  """Serialize a bibtex to string representation."""
  string = bibtexparser.dumps(bibtex)
  # Strip non-ASCII characters in serialized bibtex since LaTeX complains about
  # "Unicode char X not set up for use with LaTeX."
  string = string.encode("ascii", "ignore").decode("ascii")
  string = string.replace("\u200B", "")
  return string


def main():
  """Main entry point."""

  # Read the input.
  with open(FLAGS.bibtex_path) as f:
    bibtex = bibtexparser.load(f)

  ExpandBibtexShorthandInPlace(bibtex, expansions)
  string = BibtexToString(bibtex)

  # Write the result.
  with open(FLAGS.bibtex_path, "w") as f:
    f.write(string)


if __name__ == "__main__":
  app.Run(main)
