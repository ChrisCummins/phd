#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of DeepSmith.
#
# DeepSmith is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# DeepSmith is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# DeepSmith.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Parser for dsmith
"""
import datetime
import logging
import re
import sys

from collections import namedtuple

import dsmith


__help__ = f"""\
This is the DeepSmith interactive prompt.
"""


class UnrecognizedInput(ValueError):
    pass


class Parsed(object):
    def __init__(self):
        self.msg = None
        self.func = None
        self.args = []
        self.kwargs = {}


def _help_func(*args, **kwargs):
    file = kwargs.pop("file", sys.stdout)
    print(__help__, file=file)

def _exit_func(*args, **kwargs):
    file = kwargs.pop("file", sys.stdout)
    print("Have a nice day!", file=file)
    sys.exit()


def parse(statement: str) -> Parsed:
    if not isinstance(statement, str): raise TypeError

    parsed = Parsed()

    # parsing is case insensitive
    statement = re.sub("\s+", " ", statement.strip().lower())
    components = statement.split(" ")

    csv = ", ".join(f"'{x}'" for x in components)
    logging.debug(f"parsing input [{csv}]")

    if not statement:
        return parsed

    if len(components) == 1 and re.match(r'(hi|hello|hey)', components[0]):
        parsed.msg = "Hi there!"
    elif len(components) == 1 and re.match(r'(exit|quit)', components[0]):
        parsed.func = _exit_func
    elif len(components) < 3 and components[0] == "help":
        parsed.func = _help_func
        parsed.args = components[1:]
    else:
        raise UnrecognizedInput("I'm sorry, I don't understand. "
                                "Type 'help' for available commands.")

    return parsed


def repl(file=sys.stdout) -> None:
    hour = datetime.datetime.now().hour

    greeting = "Good evening."
    if hour > 4:
        greeting = "Good morning."
    if hour > 12 and hour < 18:
        greeting = "Good afternoon."

    print(greeting, file=file)

    try:
        while True:
            sys.stdout.write("> ")
            choice = input()

            try:
                parsed = parse(choice)

                if parsed.msg:
                    print(parsed.msg, file=file)

                if parsed.func:
                    args = ", ".join(f"'{x}'" for x in parsed.args)
                    kwargs = ""
                    debug_msg = f"func = {parsed.func.__name__}, args = [{args}], kwargs = {{{kwargs}}}"

                    logging.debug(debug_msg)
                    parsed.func(*parsed.args, **parsed.kwargs)
            except UnrecognizedInput as e:
                print(e, file=file)

    except KeyboardInterrupt:
        print("", file=file)
        _exit_func(file=file)
