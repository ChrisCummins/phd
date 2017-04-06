import sys

from argparse import ArgumentParser
from pkg_resources import require
from typing import Tuple

from cldrive import *


def parse_ndrange(ndrange: str) -> Tuple[int, int, int]:
    components = ndrange.split(',')
    assert(len(components) == 3)
    return (int(components[0]), int(components[1]), int(components[2]))


def print_version_and_exit() -> None:
    version = require("cldrive")[0].version
    print(f"cldrive {version}")
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

        if len(args) == 2 and args[0] == "--porcelain":
            porcelain(args[1])

        # parse args normally
        return super(CliParser, self).parse_args(args, namespace)
