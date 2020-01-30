# Copyright 2018-2020 Chris Cummins <chrisc.101@gmail.com>
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
import os

import gspread
from gspread.exceptions import WorksheetNotFound
from oauth2client.service_account import ServiceAccountCredentials


def csv_to_worksheet(worksheet, list_of_lists):
  """ set worksheet values from list of lists """
  nrows = len(list_of_lists)
  assert nrows
  ncols = len(list_of_lists[0])
  for l in list_of_lists[1:]:
    assert len(l) == ncols

  worksheet.resize(nrows, ncols)

  cell_list = worksheet.range(1, 1, nrows, ncols)

  for i, cell in enumerate(cell_list):
    cell.value = list_of_lists[i // ncols][i % ncols]

  worksheet.update_cells(cell_list)


def get_or_create_spreadsheet(gc, name, share_with):
  """ return spreadsheet by name, creating it if necessary """
  try:
    sh = gc.open(name)
  except SpreadsheetNotFound:
    sh = gc.create(name)
    sh.share(share_with, perm_type="user", role="writer")
  return sh


def get_or_create_worksheet(sh, name):
  """ return worksheet by name, creating it if necessary """
  try:
    return sh.worksheet(name)
  except WorksheetNotFound:
    return sh.add_worksheet(title=name, rows=1, cols=1)


def get_connection(keypath):
  scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
  ]

  credentials = ServiceAccountCredentials.from_json_keyfile_name(
    os.path.expanduser(keypath), scope
  )

  return gspread.authorize(credentials)
