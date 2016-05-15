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

def create_corpus(db, out_path):
    # Dump all of the rewritten OpenCL files, ordered by the number of
    # stargazers of the file repo.
    print('creating training corpus', out_path, '...')

    c = db.cursor()
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


def main():
    parser = ArgumentParser()
    parser.add_argument('input', help='path to SQL input dataset')
    args = parser.parse_args()

    db_path = args.input

    db = sqlite3.connect(db_path)
    create_corpus(db, 'input.txt')


if __name__ == '__main__':
    main()
