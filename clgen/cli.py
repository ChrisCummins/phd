#
# Copyright 2016 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Command line interface to clgen.
"""
import json
import os

from argparse import ArgumentParser
from sys import argv, exit

import clgen
from clgen import log


def print_version_and_exit():
    """
    Print the clgen version. This function does not return.
    """
    print("clgen version", clgen.version())
    exit(0)


def getparser():
    """
    Get the clgen argument parser.

    Returns:
        ArgumentParser: clgen cli argument parser
    """
    parser = ArgumentParser(
        description="Generate OpenCL programs using Deep Learning.")
    parser.add_argument("model_json", metavar="<model-json>",
                        help="path to model specification file")
    parser.add_argument("arguments_json", metavar="<arguments-json>",
                        help="path to arguments specification file")
    parser.add_argument("sample_json", metavar="sample-json", nargs="?",
                        help="path to sample specification file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("--version", action="store_true",
                        help="show version information and exit")
    return parser


def load_json_file(path):
    try:
        with open(clgen.must_exist(path)) as infile:
            return json.loads(infile.read())
    except ValueError as e:
        log.fatal("malformed file '{}'. Message from parser: ".format(
                      os.path.basename(path)),
                  "    " + str(e),
                  "Hope that makes sense!", sep="\n")
    except clgen.File404:
        log.fatal("could not find file '{}'".format(path))

def main(*argv):
    """
    Main entry point to clgen command line interface.

    This function does not return.

    Args:
        *argv (str): Command line arguments.
    """
    # --version option overrides the normal argument parsing process.
    if len(argv) == 1 and argv[0] == "--version":
        print_version_and_exit()

    # Parse arguments. Will not return if arguments are bad, or -h|--help flag.
    parser = getparser()
    args = parser.parse_args(argv)

    if args.version:
        print_version_and_exit()

    # Start the logging engine.
    log.init(args.verbose)

    # Read input configuration files.
    model = load_json_file(args.model_json)
    arguments = load_json_file(args.arguments_json)
    sample = load_json_file(
        args.sample_json or clgen.data_path("default-sample.json"))

    try:
        clgen.main(model, arguments, sample)
        exit(0)
    except Exception as e:
        log.fatal(
            type(e).__name__ + ":", e,
            "\n\nPlease report bugs to Chris Cummins <chrisc.101@gmail.com>")
