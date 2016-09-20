"""
Command line interface to clgen.
"""
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

    exit(0)
