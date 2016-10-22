# Copyright (C) 2015, 2016 Chris Cummins.
#
# This file is part of labm8.
#
# Labm8 is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Labm8 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with labm8.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Wrapper for invoking Makefiles.
"""
import os
import re
import subprocess

import labm8
from labm8 import fs
from labm8 import system


class Error(Exception):
    """
    Module-level error class.
    """
    pass


class MakeError(Error):
    """
    Thrown if make command fails.
    """
    pass


class NoMakefileError(Error):
    """
    Thrown if a directory does not contain a Makefile.
    """
    pass


class NoTargetError(Error):
    """
    Thrown if there is no rule for the requested target.
    """
    pass


_BAD_TARGET_RE = re.compile(r"No rule to make target "
                            "'.+'.  Stop.")


def make(target="all", dir=".", **kwargs):
    """
    Run make.

    Arguments:

        target (str, optional): Name of the target to build. Defaults
          to "all".
        dir (str, optional): Path to directory containing Makefile.
        **kwargs (optional): Any additional arguments to be passed to
          system.run().

    Returns:

        (int, str, str): The first element is the return code of the
          make command. The second and third elements are the stdout
          and stderr of the process.

    Raises:

        NoMakefileError: In case a Makefile is not found in the target
          directory.
        NoTargetError: In case the Makefile does not support the
          requested target.
        MakeError: In case the target rule fails.
    """
    if not fs.isfile(fs.path(dir, "Makefile")):
        raise NoMakefileError("No makefile in '{}'".format(fs.abspath(dir)))

    fs.cd(dir)

    # Default parameters to system.run()
    if "timeout" not in kwargs: kwargs["timeout"] = 300

    ret, out, err = system.run(["make", target], **kwargs)
    fs.cdpop()

    if ret > 0:
        if re.search(_BAD_TARGET_RE, err):
            raise NoTargetError("No rule for target '{}'"
                                .format(target))
        else:
            raise MakeError("Target '{}' failed".format(target))

        raise MakeError("Failed")

    return ret, out, err


def clean(**kwargs):
    """
    Run make clean.

    Equivalent to invoking make() with target="clean".

    Arguments:

        **kwargs (optional): Any additional arguments to be passed to
          make().

    Returns:

        (int, str, str): The first element is the return code of the
          make command. The second and third elements are the stdout
          and stderr of the process.

    Raises:

        NoMakefileError: In case a Makefile is not found in the target
          directory.
        NoTargetError: In case the Makefile does not support the
          requested target.
        MakeError: In case the target rule fails.
    """
    make(target="clean", **kwargs)
