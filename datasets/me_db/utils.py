# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Utility code for me.db."""
import csv
import datetime
import os

from labm8.py import app

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
