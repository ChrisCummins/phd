#!/usr/bin/env python3.6

import logging

from argparse import ArgumentParser, FileType

import me
import me.healthkit
import me.omnifocus
import me.aggregate


def main():
    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable more verbose logging output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s")

    with open("omnifocus.json") as infile:
        me.omnifocus.process_json(infile, "csv")

    with open("/Users/cec/Documents/me.csv/export.zip") as infile:
        me.healthkit.process_archive(infile, "csv")

    me.aggregate.aggregate("csv")


if __name__ == "__main__":
    main()