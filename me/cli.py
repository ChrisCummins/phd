#!/usr/bin/env python3.6

import json
import logging
import os
import sys

from argparse import ArgumentParser, FileType
from tempfile import TemporaryDirectory

import me
import me.healthkit
import me.omnifocus
import me.aggregate


def get_config(path):
    """ read config file """
    with open(path) as infile:
        data = json.load(infile)
    return data


def init_logging(verbose: bool=False):
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

    # Sources options:
    healthkit_path = os.path.expanduser(config["sources"]["healthkit"]["export_path"])
    of2_path = os.path.expanduser(config["sources"]["omnifocus"]["of2_path"])

    # Export options:
    csv_path = os.path.expanduser(config["exports"]["csv"]["path"])
    spreadsheet_name = config["exports"]["gsheet"]["name"]
    keypath = os.path.expanduser(config["exports"]["gsheet"]["keypath"])
    share_with = config["exports"]["gsheet"]["share_with"]

    # Create and process OmniFocus data:
    me.omnifocus.export_csvs(of2_path, csv_path)

    # Process Healthkit data:
    with open(healthkit_path) as infile:
        me.healthkit.process_archive(infile, csv_path)

    # Aggregate data:
    me.aggregate.aggregate(csv_path, spreadsheet_name, keypath, share_with)


def main():
    try:
        _main()
    except KeyboardInterrupt:
        print("interrupt", file=sys.stderr)
