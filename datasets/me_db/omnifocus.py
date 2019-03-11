import csv
import datetime
import json
import os
import subprocess
from tempfile import TemporaryDirectory

from datasets.me_db import utils as me


def count_tasks(count, data, complete: bool):
  if isinstance(data, list):
    for node in data:
      count = count_tasks(count, node, complete)
  else:
    if "completed" in data:
      count += 1 if data["completed"] == complete else 0
    for node in data:
      if isinstance(data[node], list) or isinstance(data[node], dict):
        count = count_tasks(count, data[node], complete)
  return count


def _get_tasks(tasks, data):
  if isinstance(data, list):
    for node in data:
      _get_tasks(tasks, node)
  else:
    if "dateAdded" in data and "completionDate" in data:
      if data["dateAdded"]:
        tasks.append((data["dateAdded"], data["completionDate"], data["name"]))
    for node in data:
      if isinstance(data[node], list) or isinstance(data[node], dict):
        _get_tasks(tasks, data[node])
  return tasks


def _get_task(task):
  started, completed, name = task
  if started:
    started = datetime.datetime.utcfromtimestamp(started / 1000)
  if completed:
    completed = datetime.datetime.utcfromtimestamp(completed / 1000)
  return started, completed, name


def get_tasks(data):
  tasks = _get_tasks([], data)
  tasks = [_get_task(x) for x in sorted(tasks, key=lambda x: x[0])]
  return tasks


def task_count(tasks, date, completed: bool):
  count = 0
  for task in tasks:
    # count complete and incomplete tasks:
    if completed and task[1] and task[1].date() <= date:
      count += 1
    elif not completed and task[0] and task[0].date() <= date:
      count += 1
  return count


def process_json(infile, outpath):
  me.mkdir(os.path.dirname(outpath))

  app.Debug(f"Parsing {infile.name}")
  data = json.load(infile)

  completed = count_tasks(0, data, complete=True)
  incomplete = count_tasks(1, data, complete=False)

  tasks = get_tasks(data)
  start = tasks[0][0].date()
  today = datetime.date.today()

  with open(outpath, "w") as outfile:
    writer = csv.writer(outfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)

    # Write header
    writer.writerow([
        "Date", "Incomplete Tasks", "Complete Tasks", "Tasks Added",
        "Tasks Completed", "Tasks Delta"
    ])

    last_incomplete, last_complete = 0, 0

    for date in me.daterange(start, today):
      incomplete = task_count(tasks, date, completed=False)
      complete = task_count(tasks, date, completed=True)
      delta_added = incomplete - last_incomplete
      delta_completed = complete - last_complete
      delta = delta_added - delta_completed
      writer.writerow(
          [date, incomplete, complete, delta_added, delta_completed, delta])
      last_incomplete = incomplete
      last_complete = complete

    nrows = len(tasks)
    app.Info(f"Exported {nrows} records to \"{outfile.name}\"")


def generate_json(of2path, outpath):
  app.Debug(f"Exporting OmniFocus database to JSON")
  subprocess.check_output([of2path, '-o', outpath])
  return outpath


def export_csvs(of2path, outpath):
  with TemporaryDirectory(prefix="me.csv-") as tmpdir:
    pwd = os.getcwd()
    os.chdir(tmpdir)
    jsonpath = generate_json(of2path, "omnifocus.json")
    with open(jsonpath) as infile:
      process_json(infile, f"{outpath}/Tasks.csv")
    os.chdir(pwd)
