#!/usr/bin/env python
"""Remove redundant files in bibtex entries.

Mendeley exports a bibliographies with a bunch of extra fields that I don't
want in bibtex files. This script removes the unwanted fields.
"""

import bibtexparser

from labm8 import app


FLAGS = app.FLAGS

app.DEFINE_input_path('bibtex_path', None, 'Path of bibtex file.', required=True)


def DeleteKeys(dictionary, keys):
  """Remove 'keys' from 'dictionary'."""
  for key in keys:
    if key in dictionary:
      del dictionary[key]


def main():
  """Main entry point."""

  # Read the input.
  with open(FLAGS.bibtex_path) as f:
    bibtex = bibtexparser.load(f)

  # Strip the unwanted keys from the entries.
  for entry in bibtex.entries:
    DeleteKeys(entry, [
        'abstract',
        'annote',
        'archiveprefix',
        'arxivid',
        'eprint',
        'file',
        'doi',
        'pages',
        'isbn',
        'issn',
        'keywords',
        'mendeley-tags',
        'pmid',
        'primaryclass',
        'url',
    ])

  string = bibtexparser.dumps(bibtex)
  # Strip non-ASCII characters in serialized bibtex since LaTeX complains about
  # "Unicode char X not set up for use with LaTeX."
  string = string.encode('ascii', 'ignore').decode('ascii')
  string = string.replace(u"\u200B", "")

  # Write the result.
  with open(FLAGS.bibtex_path, 'w') as f:
    f.write(string)


if __name__ == '__main__':
  app.Run(main)
