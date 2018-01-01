#!/usr/bin/env python3.6

import csv
import logging
import os
import re

from argparse import ArgumentParser, FileType
from collections import defaultdict
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from xml.dom import minidom


def process_records(typename, records, outdir):
    # build a list of attributes names (columns)
    attributes = set()
    for record in records:
        for attr in record.keys():
            attributes.add(attr)
    attributes = sorted([x for x in attributes if x != "type"])

    # Create CSV file
    with open(f"{outdir}/{typename}.csv", "w") as outfile:
        logging.debug(f"Creating CSV file {outfile.name}")
        writer = csv.writer(outfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)

        # Write header
        writer.writerow(attributes)

        # Write rows
        for record in records:
            row = []
            for attr in attributes:
                try:
                    row.append(record[attr].value)
                except:
                    row.append('')
            writer.writerow(row)

        nrows = len(records)
        logging.info(f"Exported {nrows} records for \"{typename}\"")


def process_file(infile, outdir):
    logging.debug(f"Parsing export.xml")
    xmldoc = minidom.parse(infile.name)
    recordlist = xmldoc.getElementsByTagName('Record')

    data = defaultdict(list)
    for s in recordlist:
        typename = s.attributes['type'].value
        # Strip the HealthKit prefix from the type name:
        typename = typename[len("HKQuantityTypeIdentifier"):]
        # Split the CamelCase name into separate words:
        typename = " ".join(re.findall('[A-Z][^A-Z]*', typename))

        data[typename].append(s.attributes)

    for typename in data:
        process_records(typename, data[typename], outdir)


def process_archive(infile, outdir):
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass

    with TemporaryDirectory(prefix="me.csv.") as unzipdir:
        logging.debug(f"Unpacking healthkit archive to {unzipdir}")
        archive = ZipFile(infile.name)
        archive.extract("apple_health_export/export.xml", path=unzipdir)
        with open(f"{unzipdir}/apple_health_export/export.xml") as infile:
            process_file(infile, outdir)


def main():
    parser = ArgumentParser()
    parser.add_argument("infile", metavar="<export.zip>", type=FileType('r'),
                        help="Path to HealthKit export")
    parser.add_argument("outdir", metavar="<dir>",
                        help="Path to output CSV files")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable more verbose logging output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s")

    process_archive(args.infile, args.outdir)


if __name__ == "__main__":
    main()
