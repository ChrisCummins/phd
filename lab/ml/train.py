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


def table_exists(db, table_name):
    c = db.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='"
              + table_name + "'")
    res = c.fetchone()
    c.close()
    return res and res[0]


def create_corpus(db, out_path, gh=False, fileid=False, reverse=False,
                  status=0, eof=False, dir=False):
    # Dump all the preprocessed OpenCL files
    print('creating DNN corpus', out_path, '...')

    order = 'ASC' if reverse else 'DESC'

    c = db.cursor()
    if gh:
        print('ordering by number of GitHub stargazers')
        c.execute('SELECT PreprocessedFiles.id,PreprocessedFiles.contents FROM PreprocessedFiles '
                  'LEFT JOIN ContentMeta ON PreprocessedFiles.id=ContentMeta.id '
                  'LEFT JOIN Repositories ON ContentMeta.repo_url=Repositories.url '
                  'WHERE PreprocessedFiles.status=' + str(status) + ' '
                  'ORDER BY Repositories.stars '
                  + order)
    else:
        print('ordering by line count')
        c.execute('SELECT id,contents FROM PreprocessedFiles '
                  'WHERE PreprocessedFiles.status=' + str(status) + ' '
                  'ORDER BY LC(contents) '
                  + order)
    rows = c.fetchall()

    if dir:
        print('writing to directory ', out_path, '/', sep='')
        if os.path.exists(out_path):
            print('fatal: directory already exists!', file=sys.stderr)
            return 1
        else:
            os.makedirs(out_path)
            for row in rows:
                id,contents = row
                path = os.path.join(out_path, id + '.txt')
                with open(path, 'w') as out:
                    out.write(contents)
            return 0
    else:
        print('writing file', out_path)
        with open(out_path, 'w') as out:
            for row in rows:
                id,contents = row
                if fileid: # Print file ID
                    out.write('/* ID: ' + id + '*/\n\n')
                out.write(contents)
                if eof: # Print EOF token
                    out.write('\n/* EOF */\n\n')
                else:
                    out.write('\n\n')
        return 0


def linecount(t):
    return len(t.split('\n'))


def main():
    parser = ArgumentParser()
    parser.add_argument('input', help='path to SQL input dataset')
    parser.add_argument('output', help='path to output file or directory')
    parser.add_argument('-d', action='store_true', default=False,
                        help='output to directory (overrides -i, --eof, -r)')
    parser.add_argument('-i', action='store_true', default=False,
                        help='include file separators')
    parser.add_argument('--eof', action='store_true', default=False,
                        help='print end of file')
    parser.add_argument('-r', action='store_true', default=False,
                        help='use reverse order')
    parser.add_argument('-s', '--status', type=int, default=0,
                        help='status code to use')
    args = parser.parse_args()

    db_path = args.input
    out_path = args.output
    opts = {
        "dir": args.d,
        "eof": args.eof,
        "fileid": args.i,
        "reverse": args.r,
        "status": args.status
    }

    db = sqlite3.connect(db_path)
    db.create_function("LC", 1, linecount)

    # auto-detect whether it's a GitHub repo
    opts['gh'] = table_exists(db, 'Repositories')

    ret = create_corpus(db, out_path, **opts)
    if ret:
        sys.exit(ret)


if __name__ == '__main__':
    main()
