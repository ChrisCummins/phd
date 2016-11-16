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
Training utils.
"""
from __future__ import with_statement
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys

from io import open
from labm8 import fs

from clgen import dbutil


def sanitize_id(id):
    """
    Sanitize ID.

    Arguments:
        id (str): ID.

    Returns:
        str: ID.
    """
    return re.sub('[/:\.]+', '-', id)


def create_corpus(db, out_path, gh=False, fileid=False, reverse=False,
                  input_samples=False, status=0, eof=False, dir=False):
    """
    Create CLgen training corpus.

    Arguments:
        db (slite3.Connection): Dataset.
        out_path (str): Path to output.
        gh (bool, optional): Dataset is GitHub.
        fileid (bool, optional): Include file IDs.
        reverse (bool, optional): Reverse ordering of output.
        input_samples (bool, optional): If True, use un-preprocessed files.
        status (int, optional): Filter preprocess status.
        eof (bool, optional): Include EOF separators.
        dir (bool, optional): Write output to directory.
    """
    # Dump all the preprocessed OpenCL files
    print('creating DNN corpus', out_path, '...')

    order = 'ASC' if reverse else 'DESC'

    c = db.cursor()

    # Query components
    table = 'ContentFiles' if input_samples else 'PreprocessedFiles'
    select = 'SELECT {}.id,{}.contents'.format(table, table, table)

    if input_samples:
        qualifier = ''
    else:
        qualifier = 'WHERE {}.status={}'.format(table, status)

    if gh:
        table += (' LEFT JOIN ContentMeta ON {}.id=ContentMeta.id'
                  ' LEFT JOIN Repositories ON '
                  'ContentMeta.repo_url=Repositories.url'
                  .format(table))
        orderby = 'Repositories.stars'
    else:
        orderby = 'LC(contents)'

    query = ('{select} FROM {table} {qualifier} ORDER BY {orderby} {order}'
             .format(select=select, table=table, qualifier=qualifier,
                     orderby=orderby, order=order))

    c.execute(query)
    rows = c.fetchall()

    if dir:
        print('writing to directory ', out_path, '/', sep='')
        if os.path.exists(out_path):
            if len(fs.ls(out_path)):
                print('fatal: directory already exists!', file=sys.stderr)
                return 1
        else:
            os.makedirs(out_path)
        for row in rows:
            id, contents = row
            path = os.path.join(out_path, sanitize_id(id) + '.cl')
            with open(path, 'w') as out:
                out.write(contents)
        return 0
    else:
        print('writing file', out_path)
        with open(out_path, 'wb') as out:
            for row in rows:
                id, contents = row
                if fileid:  # Print file ID
                    out.write('/* ID: {} */\n\n'.format(id).encode('utf-8'))
                out.write(contents.encode('utf-8'))
                if eof:  # Print EOF token
                    out.write('\n/* EOF */\n\n'.encode('utf-8'))
                else:
                    out.write('\n\n'.encode('utf-8'))
        return 0


def linecount(t):
    """
    Line count.

    Arguments:
        t (str): String.

    Returns:
        int: Line count.
    """
    return len(t.split('\n'))


def train(db_path, out_path, **kwargs):
    """
    Generate corpus.

    Arguments:
        db_path (str): Dataset.
        out_path (str): Corpus path.
        **kwargs (dict): Additional arguments to create_corpus().
    """
    db = dbutil.connect(db_path)
    db.create_function("LC", 1, linecount)

    # auto-detect whether it's a GitHub repo
    kwargs['gh'] = dbutil.is_github(db)

    ret = create_corpus(db, out_path, **kwargs)
    if ret:
        sys.exit(ret)
