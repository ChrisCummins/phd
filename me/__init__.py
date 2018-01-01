#!/usr/bin/env python3.6

import logging

from argparse import ArgumentParser, FileType
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def process_file(infile, outfile):
    logging.info("done")


def process_archive(infile, outfile):
    with TemporaryDirectory(prefix="me.csv.") as unzipdir:
        archive = ZipFile(infile.name)
        archive.extract("apple_health_export/export.xml", path=unzipdir)
        logging.debug(f"Unpacking healthkit archive to {unzipdir}")
        with open(f"{unzipdir}/apple_health_export/export.xml") as infile:
            process_file(infile, outfile)


def main():
    parser = ArgumentParser()
    parser.add_argument("infile", metavar="<export.zip>", type=FileType('r'),
                        help="Path to HealthKit export")
    parser.add_argument("outfile", metavar="<me.csv>", type=FileType('w'),
                        help="Path to output csv")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable more verbose logging output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s")

    process_archive(args.infile, args.outfile)


if __name__ == "__main__":
    main()
