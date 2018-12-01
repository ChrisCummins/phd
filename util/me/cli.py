import json
import logging
import os

import sys
from argparse import ArgumentParser

from util.me import aggregate
from util.me import healthkit
from util.me import omnifocus
from util.me import toggl_import as toggl


def get_config(path):
  """ read config file """
  with open(path) as infile:
    data = json.load(infile)
  return data


def init_logging(verbose: bool = False):
  """ set logging verbosity """
  if verbose:
    logging.getLogger().setLevel(logging.DEBUG)
  else:
    logging.getLogger().setLevel(logging.INFO)
  logging.basicConfig(format="%(message)s")


def _main():
  parser = ArgumentParser()
  parser.add_argument("-c", "--config", metavar="<path>",
                      default=os.path.expanduser("~/.me.json"),
                      help="path to config file (default: ~/.me.json)")
  parser.add_argument("-v", "--verbose", action="store_true",
                      help="enable more verbose logging output")
  args = parser.parse_args()

  # Initialize logging engine:
  init_logging(args.verbose)

  # Get and parse config file:
  config = get_config(args.config)

  # Export options:
  csv_path = os.path.expanduser(config["exports"]["csv"]["path"])
  spreadsheet_name = config["exports"]["gsheet"]["name"]
  google_keypath = os.path.expanduser(config["exports"]["gsheet"]["keypath"])
  share_with = config["exports"]["gsheet"]["share_with"]

  # Sources options:
  healthkit_config = config["sources"]["healthkit"]
  healthkit_path = os.path.expanduser(healthkit_config["export_path"])
  healthkit_csvs = [
    f"{csv_path}/HealthKit/{x}.csv" for x in healthkit_config["aggregate"]]

  toggl_config = config["sources"]["toggl"]
  toggl_keypath = os.path.expanduser(toggl_config["keypath"])
  toggl_workspace = toggl_config["workspace"]
  toggl_start = toggl_config["start_date"]
  toggl_csvs = [f"{csv_path}/Toggl/{x}.csv" for x in toggl_config["aggregate"]]

  omnifocus_config = config["sources"]["omnifocus"]
  of2_path = os.path.expanduser(omnifocus_config["of2_path"])
  omnifocus_csvs = [f"{csv_path}/OmniFocus/Tasks.csv"]

  # Create and process OmniFocus data:
  omnifocus.export_csvs(of2_path, f"{csv_path}/OmniFocus")

  # Process Healthkit data:
  with open(healthkit_path) as infile:
    healthkit.process_archive(infile, f"{csv_path}/HealthKit")

  # Create and process Toggl data:
  toggl.export_csvs(f"{csv_path}/Toggl", toggl_keypath, toggl_workspace,
                    toggl_start)

  # Aggregate data:
  csvs = toggl_csvs + healthkit_csvs + omnifocus_csvs
  aggregate.aggregate(csv_path, csvs, spreadsheet_name, google_keypath,
                      share_with)


def main():
  try:
    _main()
  except KeyboardInterrupt:
    print("interrupt", file=sys.stderr)
