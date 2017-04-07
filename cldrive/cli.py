import sys

from argparse import ArgumentParser
from typing import Tuple

from cldrive import *


def print_version_and_exit() -> None:
    print(f"cldrive {__version__}")
    sys.exit(0)


class CliParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        """
        See python argparse.ArgumentParser.__init__().
        """
        # append author information to description
        description = kwargs.get("description", "")

        if len(description) and description[-1] != "\n":
            description += "\n"
        description += """
Copyright (C) 2017 Chris Cummins <chrisc.101@gmail.com>.
<https://github.com/ChrisCummins/cldrive>"""

        kwargs["description"] = description.lstrip()

        # call built in ArgumentParser constructor.
        super(CliParser, self).__init__(*args, **kwargs)

        # Add defualt arguments
        self.add_argument("--version", action="store_true",
                          help="show version information and exit")


    def parse_args(self, args=sys.argv[1:], namespace=None):
        """
        See python argparse.ArgumentParser.parse_args().
        """
        # intercept args early:
        if "--version" in args:
            print_version_and_exit()

        # parse args normally
        return super(CliParser, self).parse_args(args, namespace)
