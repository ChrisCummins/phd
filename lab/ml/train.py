#!/usr/bin/env python3
#
# Train machine learning models on dataset.
#
# TODO:
#
# ML:
#
#   Train char-rnn on raw text
#   Train char-rnn on files which compiled
#   Train char-rnn on "cleaned" text (no comments, etc)
#   Train char-rnn on bytecode
#   Train char-rnn on AST(?)

import os
from collections import defaultdict

try:
    import cPickle as pickle
except ImportError:
    import pickle

from pymarkovchain import MarkovChain

db = 'markov.db'
src = 'cl.txt'
dst = 'gen.txt'

num_lines = 10000


def _db_factory():
    return defaultdict(_one_dict)

def _one():
    return 1.0

def _one_dict():
    return defaultdict(_one)

def main():

    # Hack to get MarkovChain to run:
    if not os.path.exists(db):
        with open(db, 'wb') as out:
            pickle.dump(_db_factory(), out)

    mc = MarkovChain(db)

    print('training model ...')
    with open(src, 'r') as infile:
        mc.generateDatabase(infile.read())

    # To let the markov chain generate some text, execute:
    print('generating text ...')
    with open(dst, 'w') as out:
        i = 0
        while i < num_lines:
            out.write(mc.generateString() + '\n')
            i += 1


if __name__ == '__main__':
    main()
