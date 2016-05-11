#!/usr/bin/env python3
#
# Generate OpenCL programs using clsmith.
#
import json
import os
import re
import sys
import sqlite3

from argparse import ArgumentParser
from hashlib import sha1
from subprocess import Popen,PIPE,STDOUT

# Counters
files_new_counter = 0
errors_counter = 0


class CLSmithException(Exception): pass
class ShaException(Exception): pass
class HeaderNotFoundException(Exception): pass


def print_counters():
    print('\r\033[Kfiles: new ', files_new_counter,
          '. errors ', errors_counter,
          sep='', end='')
    sys.stdout.flush()


_include_re = re.compile('\w*#include ["<](.*)[">]')


def include_path(name):
    dirs = ('~/phd/extern/clsmith/runtime',
            '~/phd/extern/clsmith/build')
    for dir in dirs:
        path = os.path.join(os.path.expanduser(dir), name)
        if os.path.exists(path):
            return path
    raise HeaderNotFoundException(name)


def inline_headers(src):
    outlines = []
    for line in src.split('\n'):
        match = re.match(_include_re, line)
        if match:
            include_name = match.group(1)

            path = include_path(include_name)
            with open(path) as infile:
                header = infile.read()
                outlines.append(inline_headers(header))
        else:
            outlines.append(line)

    return '\n'.join(outlines)


def get_new_program(db_path):
    global files_new_counter

    clsmith = os.path.expanduser('~/phd/extern/clsmith/build/CLSmith')
    outputpath = 'CLProg.c'

    db = sqlite3.connect(db_path)
    c = db.cursor()

    cmd = [clsmith]

    process = Popen(cmd)
    process.communicate()

    if process.returncode != 0:
        raise CLSmithException()

    with open(outputpath) as infile:
        contents = infile.read()

    contents = inline_headers(contents)

    sha = sha1(contents.encode('utf-8')).hexdigest()

    c.execute('INSERT INTO ContentFiles VALUES(?,?)',
              (sha, contents))
    db.commit()
    db.close()
    files_new_counter += 1
    print_counters()


def main():
    global errors_counter

    parser = ArgumentParser()
    parser.add_argument('input', help='path to SQL input dataset')
    parser.add_argument('-n', type=int, default=5000,
                        help='number of OpenCL kernels to generate')
    args = parser.parse_args()
    db_path = args.input
    target_num_kernels = args.n

    print('generating', args.n, 'kernels to', args.input)

    db = sqlite3.connect(db_path)
    c = db.cursor()
    c.execute('SELECT Count(*) FROM ContentFiles')
    num_kernels = c.fetchone()[0]
    while num_kernels < target_num_kernels:
        get_new_program(db_path)
        c.execute('SELECT Count(*) FROM ContentFiles')
        num_kernels = c.fetchone()[0]

    print_counters()
    print("\n\ndone.")
    db.close()


if __name__ == '__main__':
    main()
