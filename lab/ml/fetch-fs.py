#!/usr/bin/env python3
#
# Search for OpenCL files in a directory.
#
import re
import sys
import sqlite3
import os

from argparse import ArgumentParser

_include_re = re.compile('\w*#include ["<](.*)[">]')

def inline_headers(path, stack):
    stack.append(path)

    with open(path) as infile:
        src = infile.read()

    outlines = []
    for line in src.split('\n'):
        match = re.match(_include_re, line)
        if match:
            include_name = match.group(1)

            # Try and resolve relative paths
            include_name = include_name.replace('../', '')

            include_path = os.path.join(os.path.dirname(path), include_name)

            if os.path.exists(include_path) and include_path not in stack:
                include_src = inline_headers(include_path, stack)
                outlines.append('// [FETCH] include: ' + include_path)
                outlines.append(include_src)
                outlines.append('// [FETCH] eof(' + include_path + ')')
            else:
                if include_path in stack:
                    outlines.append('// [FETCH] ignored recursive include: ' + include_path)
                else:
                    outlines.append('// [FETCH] 404 not found: ' + include_path)
        else:
            outlines.append(line)

    return '\n'.join(outlines)


def process_cl_file(db_path, path):
    db = sqlite3.connect(db_path)
    c = db.cursor()

    contents = inline_headers(path, [])
    c.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)', (path,contents))

    db.commit()
    c.close()


def is_opencl_path(path):
    return path.endswith('.cl') or path.endswith('.ocl')


def get_cl_paths(root):
    return [os.path.join(root, f) for root, dn, fn in
            os.walk(root) for f in fn if is_opencl_path(os.path.join(root, f))]

def main():
    global errors_counter

    parser = ArgumentParser()
    parser.add_argument('input', help='path to SQL dataset')
    parser.add_argument('path', help='path to directory with OpenCL kernels')
    args = parser.parse_args()
    db_path = os.path.expanduser(args.input)
    dir_path = os.path.expanduser(args.path)

    files = get_cl_paths(dir_path)
    for path in files:
        process_cl_file(db_path, path)

    print("\r\033[K\ndone.")


if __name__ == '__main__':
    main()
