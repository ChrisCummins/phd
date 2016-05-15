#!/usr/bin/env python3
#
# Train machine learning models on dataset.
#
#   Include only preprocessed files in training set with unique md5s
#
#   ML:
#
#     Train char-rnn on raw text
#     Train char-rnn on files which compiled
#     Train char-rnn on "cleaned" text (no comments, etc)
#     Train char-rnn on bytecode
#     Train char-rnn on AST(?)
import sys
import os
import sqlite3

from argparse import ArgumentParser
from collections import defaultdict

try:
    import cPickle as pickle
except ImportError:
    import pickle

from pymarkovchain import MarkovChain

markov_db = 'markov.db'
dst = 'gen.txt'

num_lines = 10000

def _db_factory():
    return defaultdict(_one_dict)

def _one():
    return 1.0

def _one_dict():
    return defaultdict(_one)


def dump_training_data(db):
    print('dump training data ...')
    out_path = 'input.txt'

    c = db.cursor()
    # Get all of the rewritten OpenCL files, ordered by the number of
    # stars in the containing repo.
    c.execute('SELECT PreprocessedFiles.contents FROM PreprocessedFiles '
              'LEFT JOIN ContentMeta ON PreprocessedFiles.id=ContentMeta.id '
              'LEFT JOIN Repositories ON ContentMeta.repo_url=Repositories.url '
              'WHERE PreprocessedFiles.status=0 ORDER BY Repositories.stars DESC')
    query = c.fetchall()

    with open(out_path, 'w') as out:
        for row in query:
            contents = row[0]
            out.write(contents)
            out.write('\n')

    return out_path


def main():
    parser = ArgumentParser()
    parser.add_argument('input', help='path to SQL input dataset')
    args = parser.parse_args()

    db_path = args.input

    db = sqlite3.connect(db_path)
    data_path = dump_training_data(db)

    # Hack to get MarkovChain to run:
    if not os.path.exists(markov_db):
        with open(markov_db, 'wb') as out:
            pickle.dump(_db_factory(), out)

    mc = MarkovChain(markov_db)

    print('training model ...')
    with open(data_path, 'r') as infile:
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
