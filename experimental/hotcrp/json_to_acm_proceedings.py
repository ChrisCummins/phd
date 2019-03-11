"""Format a CSV file for ACM proceedings.

This reads as input a JSON file as exported from HotCRP ("Download" > "JSON"),
and outputs a CSV file in the format required for ACM proceedings:

  https://www.acm.org/publications/gi-proceedings
"""
import csv
import json
import sys

from labm8 import app

FLAGS = app.FLAGS

app.DEFINE_string('input_json', None, 'Path of the HotCRP JSON file to read.')


def ReadJsonFromPath(path):
  """Read JSON data from file."""
  with open(path) as f:
    return json.load(f)


def JsonToCsv(data, output_fp, paper_type="Full Paper") -> None:
  """Convert JSON data to CSV and write.

  The output format is:
    paper_type,paper_title,authors,primary_email,secondary_emails.

  As described in https://www.acm.org/publications/gi-proceedings
  """
  writer = csv.writer(output_fp, quoting=csv.QUOTE_ALL)
  for paper in data:
    authors_str = ';'.join(f"{a['first']} {a['last']}:{a['affiliation']}"
                           for a in paper['authors'])
    primary_email = paper['authors'][0]['email']
    emails_str = ';'.join(a.get('email', '') for a in paper['authors'][1:])
    writer.writerow([
        paper_type,
        paper['title'],
        authors_str,
        primary_email,
        emails_str,
    ])


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  input_json = ReadJsonFromPath(FLAGS.input_json)
  JsonToCsv(input_json, sys.stdout)


if __name__ == '__main__':
  app.RunWithArgs(main)
