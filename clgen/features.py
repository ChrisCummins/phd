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
OpenCL feature extraction.
"""
import csv
import os
import re
import sys

from collections import OrderedDict
from io import open
from labm8 import fs
from labm8 import math as labmath
from subprocess import Popen, PIPE, STDOUT

from clgen import native


def _shim_args(use_shim=False):
    """get shim header args"""
    args = []
    if use_shim:
        args += ["-DCLGEN_FEATURES", "-include", native.SHIMFILE]
    return args


def _is_features(line):
    """true if features"""
    return len(line) == 10


def _is_good_features(line, stderr):
    """true if feature extractor worked"""
    if _is_features(line):
        has_err = False if stderr.find(' error: ') == -1 else True
        return not has_err
    return False


def features(path, file=sys.stdout, fatal_errors=False, use_shim=False,
             quiet=False):
    """
    Print CSV format features of file.

    Arguments:
        path (str): Path.
        file (pipe, optional): Target to print to.
        fatal_errors (bool, optional): Exit on error.
        use_shim (bool, optional): Inject shim header.
        quiet (bool, optional): Don't print compiler output on errors.
    """
    cmd = [native.CLGEN_FEATURES, path] + [
        '-extra-arg=' + x for x in _shim_args(use_shim=use_shim)]

    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    stdout, stderr = stdout.decode('utf-8'), stderr.decode('utf-8')

    data = [line.split(',') for line in stdout.split('\n')]

    if stderr:
        has_error = re.search(" error: ", stderr)
        if has_error:
            if quiet:
                print("error:", path, file=sys.stderr)
            else:
                print("=== COMPILER OUTPUT FOR", path, "===", file=sys.stderr)
                print(stderr, file=sys.stderr)
        if fatal_errors and has_error:
            sys.exit(1)

    if process.returncode != 0:
        print("error: compiler crashed on '{}'".format(path), file=sys.stderr)
        return

    for line in data[1:]:
        if _is_good_features(line, stderr):
            print(','.join(line), file=file)


def feature_headers(file=sys.stdout):
    """
    Print CSV format feature header.

    Arguments:
        file (pipe, optional): Target to print to.
    """
    cmd = [native.CLGEN_FEATURES, '-header-only']
    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, _ = process.communicate()
    stdout = stdout.decode('utf-8').strip()

    print(stdout, file=file)


def files(paths, header=True, file=sys.stdout, **kwargs):
    """
    Print feature values of files in CSV format.

    Arguments:
        paths (str[]): Files.
        header (bool, optional): Include CSV format header.
        file (pipe, optional): Target to print to.
        **kwargs (optional): Additional arguments to features().
    """
    npaths = len(paths)

    if header:
        feature_headers(file=file)
    for path in paths:
        print(path, file=sys.stderr)
        features(path, file=file, **kwargs)


def summarize(csv_path):
    """
    Summarize a CSV file of feature values.

    Arguments:
        csv_path (str): Path to csv.

    Returns:
        dict: Summary values.
    """
    with open(csv_path) as infile:
        reader = csv.reader(infile)
        table = [row for row in reader]

    d = OrderedDict()
    ignored_cols = 2
    d['datapoints'] = len(table) - 1
    for i, col in enumerate(table[0][ignored_cols:]):
        i += ignored_cols
        d[col] = labmath.mean([float(r[i]) for r in table[1:]])

    return d
