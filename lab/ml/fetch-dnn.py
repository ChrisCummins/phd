#!/usr/bin/env python3
#
# Extract OpenCL programs from a DNN sample.
#
import re
import sys
import sqlite3
import os

from argparse import ArgumentParser
from hashlib import sha1

# Counters
kernel_counter = 0


def get_substring_idxs(substr, s):
    return [m.start() for m in re.finditer(substr, s)]


def get_cl_kernel(s, start_idx):
    global kernel_counter
    kernel_counter += 1
    print('\r\033[Kkernel:', kernel_counter, end='')
    sys.stdout.flush()

    i = s.find('{', start_idx) + 1
    d = 1  # depth
    while i < len(s) and d > 0:
        if s[i] == '{':
            d += 1
        elif s[i] == '}':
            d -= 1
        i += 1

    return s[start_idx:i]


def get_cl_kernels(s):
    idxs = get_substring_idxs('__kernel void ', s)
    print('extracting', len(idxs), 'kernels ...')
    kernels = [get_cl_kernel(s, i) for i in idxs]
    print()
    return kernels


def checksum(s):
    return sha1(s.encode('utf-8')).hexdigest()


def process_sample_file(db_path, sample_path, first_only=False):
    db = sqlite3.connect(db_path)
    c = db.cursor()

    with open(sample_path) as infile:
        sample = infile.read()
        if first_only:
            # If first_only argument is set, then only extract a
            # kernel starting at the beginning of the file.
            #
            kernels = [get_cl_kernel(sample, 0)]
        else:
            kernels = get_cl_kernels(sample)

    ids = [checksum(kernel) for kernel in kernels]

    for id,kernel in zip(ids, kernels):
        c.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)',
                  (id,kernel))
    db.commit()
    c.close()


def main():
    global errors_counter

    parser = ArgumentParser()
    parser.add_argument('input', help='path to SQL dataset')
    parser.add_argument('-d', type=str, help='path to samples directory')
    parser.add_argument('-f', type=str, default="sample.txt",
                        help='path to sample file')
    parser.add_argument('--first', action='store_true', default=False,
                        help='extract only first kernel from sample file(s)')
    args = parser.parse_args()
    db_path = args.input
    samples_dir = args.d
    sample_path = args.f
    first_only = args.first

    if samples_dir:
        files = [os.path.join(samples_dir, f) for f in os.listdir(samples_dir)
                 if os.path.isfile(os.path.join(samples_dir, f))]
        for sample_path in files:
            process_sample_file(db_path, sample_path, first_only=first_only)
    else:
        process_sample_file(db_path, sample_path, first_only=first_only)

    print("\r\033[K\ndone.")


if __name__ == '__main__':
    main()
