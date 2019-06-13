"""Remove redundant files in bibtex entries.

Mendeley exports a bibliographies with a bunch of extra fields that I don't
want in bibtex files. This script removes the unwanted fields.
"""

import bibtexparser

from labm8 import app

FLAGS = app.FLAGS

app.DEFINE_input_path('bibtex_path', None,
                      'Path of bibtex file to minimize in place')


def DeleteKeysInPlace(dictionary, keys):
  """Remove 'keys' from 'dictionary'."""
  for key in keys:
    if key in dictionary:
      del dictionary[key]


def MinimizeBibtexInPlace(bibtex) -> None:
  """Minimize a bibtex in place."""
  # Strip the unwanted keys from the entries.
  for entry in bibtex.entries:
    DeleteKeysInPlace(entry, [
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


def BibtexToString(bibtex) -> str:
  """Serialize a bibtex to string representation."""
  string = bibtexparser.dumps(bibtex)
  # Strip non-ASCII characters in serialized bibtex since LaTeX complains about
  # "Unicode char X not set up for use with LaTeX."
  string = string.encode('ascii', 'ignore').decode('ascii')
  string = string.replace(u"\u200B", "")
  return string


def main():
  """Main entry point."""

  # Read the input.
  with open(FLAGS.bibtex_path) as f:
    bibtex = bibtexparser.load(f)

  MinimizeBibtexInPlace(bibtex)
  string = BibtexToString(bibtex)

  # Write the result.
  with open(FLAGS.bibtex_path, 'w') as f:
    f.write(string)


if __name__ == '__main__':
  app.Run(main)
