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
import csv
import datetime
from collections import defaultdict

from toggl import TogglPy

from datasets.me_db import utils as me


def get_toggl(api_path):
  with open(api_path) as infile:
    api_key = infile.read().strip()
  toggl = TogglPy.Toggl()
  toggl.setAPIKey(api_key)
  return toggl


def get_workspace(toggl, name):
  for workspace in toggl.getWorkspaces():
    if workspace["name"] == name:
      return workspace
  raise LookupError(f"No Toggl Workspace with name {name}")


def get_report(toggl, params):
  """ returns a list of time entry data """

  def _get_report(records, params, page):
    """ return de-paginated list of time entries """
    params["page"] = page
    response = toggl.getDetailedReport(params)
    records += response["data"]
    if len(records) < response["total_count"]:
      records += _get_report(records, params, page + 1)
    return records

  return _get_report([], params, 1)


def parse_datetime(string):
  string = string[:-3] + string[-2:]
  return datetime.datetime.strptime(string, "%Y-%m-%dT%H:%M:%S%z")


def parse_date(string):
  return parse_datetime(string).date()


def parse_record(record):
  """ returns a (date, project, duration) tuple from a record """
  date = parse_date(record["end"])
  project = record["project"]
  duration = parse_datetime(record["end"]) - parse_datetime(record["start"])
  return (date, project, duration)


def sum_times_by_day(records):
  rows = []
  last_date = None
  aggr = 0
  for t in records:
    date, value = t
    if date != last_date:
      if last_date:
        rows.append((last_date, aggr))
      last_date = date
      aggr = value
    else:
      aggr += value
  rows.append((last_date, aggr))

  return rows


def aggregate_by_project(records):
  data = defaultdict(list)
  for t in records:
    date, project, duration = t
    data[project].append((date, duration))
  return data


def aggregate_by_project_and_day(records):
  records = aggregate_by_project(records)
  data = {}
  for project in records:
    aggr = sum_times_by_day(records[project])
    data[project] = aggr
  return data


def export_csvs(outpath, keypath, workspace_name, start_date):
  me.mkdir(outpath)

  toggl = get_toggl(keypath)
  workspace = get_workspace(toggl, workspace_name)
  records = get_report(
    toggl,
    {
      "workspace_id": workspace["id"],
      "since": start_date,
      "until": str(datetime.datetime.now().date()),  # today
    },
  )

  # Created a sorted list of tuples by date:
  tuples = sorted([parse_record(x) for x in records], key=lambda x: x[0])
  data = aggregate_by_project_and_day(tuples)

  # Export one CSV per project:
  for project in data:
    with open(f"{outpath}/{project}.csv", "w") as outfile:
      writer = csv.writer(outfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)

      writer.writerow(("Date", project))
      for row in data[project]:
        writer.writerow(row)

      nrows = len(data[project])
      app.Log(1, f'Exported {nrows} records to "{outfile.name}"')
