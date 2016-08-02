import csv
import os
import re
import sys

from collections import OrderedDict
from subprocess import Popen,PIPE,STDOUT

import labm8
from labm8 import math as labmath

import smith


def extra_args():
    return []


def is_features(line):
    return len(line) == 10


def is_good_features(line, stderr):
    if is_features(line):
        has_err = False if stderr.find(' error: ') == -1 else True
        return not has_err
    return False


def features(path, file=sys.stdout):
    features_bin = os.path.expanduser('~/phd/src/smith/native/features')
    ld_path = os.path.expanduser('~/phd/tools/llvm/build/lib/')

    cmd = [features_bin, path] + ['-extra-arg=' + x for x in extra_args()]

    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE,
                    env = {'LD_LIBRARY_PATH': ld_path})
    stdout, stderr = process.communicate()
    stdout, stderr = stdout.decode('utf-8'), stderr.decode('utf-8')

    data = [line.split(',') for line in stdout.split('\n')]

    if not len(data[0]):
        # Bad output, did it crash?
        return

    for line in data[1:]:
        if is_good_features(line, stderr):
            print(','.join(line), file=file)


def feature_headers(file=sys.stdout):
    features_bin = os.path.expanduser('~/phd/src/smith/native/features')
    ld_path = os.path.expanduser('~/phd/tools/llvm/build/lib/')

    cmd = [features_bin, '-header-only']
    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE,
                    env = {'LD_LIBRARY_PATH': ld_path})
    stdout, _ = process.communicate()
    stdout = stdout.decode('utf-8').strip()

    print(stdout, file=file)


def files(paths, file=sys.stdout):
    npaths = len(paths)

    feature_headers()
    for path in paths:
        # print("\r\033[1Kfile {} of {}".format(i, npaths), end='')
        features(path)


def summarize(csv_path):
    with open(csv_path) as infile:
        reader = csv.reader(infile)
        table = [row for row in reader]

    d = OrderedDict()
    ignored_cols = 2
    d['datapoints'] = len(table) - 1
    for i,col in enumerate(table[0][ignored_cols:]):
        i += ignored_cols
        d[col] = labmath.mean([float(r[i]) for r in table[1:]])

    return d
