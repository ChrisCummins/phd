"""Utility code for me.db."""
import csv
import datetime
import os

from labm8 import app

FLAGS = app.FLAGS


def daterange(start_date, end_date, reverse=False):
  """ returns an iterator over the specified date range """
  if reverse:
    for n in range(int((end_date - start_date).days), -1, -1):
      yield start_date + datetime.timedelta(n)
  else:
    for n in range(int((end_date - start_date).days) + 1):
      yield start_date + datetime.timedelta(n)


def mkdir(path):
  """ make directory if it does not already exist """
  try:
    os.mkdir(path)
  except FileExistsError:
    pass


def create_csv(rows, outpath):
  with open(outpath, "w") as outfile:
    app.Log(1, f"Creating CSV file {outfile.name}")

    writer = csv.writer(outfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in rows:
      writer.writerow(row)

  nrows = len(rows) - 1
  app.Log(1, f"Exported {nrows} records to '{outpath}'")
