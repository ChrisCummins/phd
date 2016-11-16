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
Fetch OpenCL files
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import json
import os
import re
import requests
import sys
import sqlite3

from base64 import b64decode
from functools import partial
from github import Github, GithubException
from hashlib import sha1
from io import open
from labm8 import fs
from subprocess import Popen
from time import sleep

import clgen
from clgen import clutil
from clgen import dbutil
from clgen import log
from clgen import explore


class FetchError(clgen.CLgenError):
    pass


# Counters
repos_new_counter = 0
repos_modified_counter = 0
repos_unchanged_counter = 0
files_new_counter = 0
files_modified_counter = 0
files_unchanged_counter = 0
errors_counter = 0
status_string = ''


def print_repo_details(repo):
    print('url:', repo.url)
    print('owner:', repo.owner.email)
    print('name:', repo.name)
    print('fork:', repo.fork)
    print('stars:', repo.stargazers_count)
    print('contributors:', len([x for x in repo.get_contributors()]))
    print('forks:', repo.forks)
    print('created_at:', repo.created_at)
    print('updated_at:', repo.updated_at)


def print_file_details(file):
    print('url:', file.url)
    print('path:', file.path)
    print('sha:', file.sha)
    print('size:', file.size)


def print_counters():
    print('\r\033[Kfiles: new ', files_new_counter,
          ', modified ', files_modified_counter,
          '. errors ', errors_counter,
          '. ', status_string[0:25],
          sep='', end='')
    sys.stdout.flush()


def rate_limit(g):
    global status_string
    remaining = g.get_rate_limit().rate.remaining
    while remaining < 100:
        sleep(1)
        status_string = 'WAITING ON RATE LIMIT'
        print_counters()
        remaining = g.get_rate_limit().rate.remaining


def process_repo(g, db, repo):
    global repos_new_counter
    global repos_modified_counter
    global repos_unchanged_counter
    global status_string

    rate_limit(g)
    url = repo.url
    updated_at = str(repo.updated_at)
    name = repo.name
    status_string = name
    print_counters()

    c = db.cursor()
    c.execute("SELECT updated_at FROM Repositories WHERE url=?", (url,))
    cached_updated_at = c.fetchone()

    # Do nothing unless updated timestamps don't match
    if cached_updated_at and cached_updated_at[0] == updated_at:
        repos_unchanged_counter += 1
        return False

    owner = repo.owner.email
    fork = 1 if repo.fork else 0
    stars = repo.stargazers_count
    try:
        contributors = len([x for x in repo.get_contributors()])
    except GithubException:
        contributors = -1
    forks = repo.forks
    created_at = repo.created_at
    updated_at = repo.updated_at

    c.execute("DELETE FROM Repositories WHERE url=?", (url,))
    c.execute("INSERT INTO Repositories VALUES(?,?,?,?,?,?,?,?,?)",
              (url, owner, name, fork, stars, contributors, forks, created_at,
               updated_at))

    if cached_updated_at:
        repos_modified_counter += 1
    else:
        repos_new_counter += 1
    db.commit()
    return True


def is_opencl_path(path):
    return path.endswith('.cl') or path.endswith('.ocl')

_include_re = re.compile('\w*#include ["<](.*)[">]')


def download_file(github_token, repo, url, stack):
    # Recursion stack
    stack.append(url)

    response = json.loads(requests.get(
        url,
        headers={
            'Authorization': 'token ' + str(github_token)
        }
    ).content.decode('utf-8'))
    src = b64decode(response['content']).decode('utf-8')

    outlines = []
    for line in src.split('\n'):
        match = re.match(_include_re, line)
        if match:
            include_name = match.group(1)

            # Try and resolve relative paths
            include_name = include_name.replace('../', '')

            branch = repo.default_branch
            tree_iterator = repo.get_git_tree(branch, recursive=True).tree
            include_url = ''
            for f in tree_iterator:
                if f.path.endswith(include_name):
                    include_url = f.url
                    break

            if include_url and include_url not in stack:
                include_src = download_file(github_token, repo, include_url)
                outlines.append(include_src)
            else:
                if not include_url:
                    outlines.append('// [FETCH] didnt find: ' + line)
                else:
                    outlines.append('// [FETCH] skipped: ' + line)
        else:
            outlines.append(line)

    return '\n'.join(outlines)


def process_file(g, github_token, db, repo, file):
    global files_new_counter
    global files_modified_counter
    global files_unchanged_counter
    global status_string

    # rate_limit(g)

    # We're only interested in OpenCL files.
    if not is_opencl_path(file.path):
        return

    url = file.url
    sha = file.sha
    path = file.path
    status_string = repo.name + '/' + path
    print_counters()

    c = db.cursor()
    c.execute("SELECT sha FROM ContentMeta WHERE id=?", (url,))
    cached_sha = c.fetchone()

    # Do nothing unless checksums don't match
    if cached_sha and cached_sha[0] == sha:
        files_unchanged_counter += 1
        return False

    repo_url = repo.url
    contents = download_file(github_token, repo, file.url, [])
    size = file.size

    c.execute("DELETE FROM ContentFiles WHERE id=?", (url,))
    c.execute("DELETE FROM ContentMeta WHERE id=?", (url,))
    c.execute("INSERT INTO ContentFiles VALUES(?,?)",
              (url, contents))
    c.execute("INSERT INTO ContentMeta VALUES(?,?,?,?,?)",
              (url, path, repo_url, sha, size))

    if cached_sha:
        files_modified_counter += 1
    else:
        files_new_counter += 1

    db.commit()
    return True


# Download all of the OpenCL on GitHub (!)
#
# Shortcomings of this appraoch:
#
#   Only includes exclusively OpenCL files, no inline strings
def github(db_path, github_username, github_pw, github_token):
    global errors_counter

    g = Github(github_username, github_pw)
    db = dbutil.connect(db_path)

    if not dbutil.is_github:
        raise clgen.UserError("not a GitHub database")

    handle_repo = partial(process_repo, g, db)

    # Fetch the repositories to iterate over. Since opencl isn't
    # treated as a first-class language by GitHub, we can't use the
    # 'language=' keyword for queries, so instead we through a much
    # wider net and filter the results afterwards.
    #
    query_terms = [
        'opencl',
        'cl',
        'khronos',
        'gpu',
        'gpgpu',
        'cuda',
        'amd',
        'nvidia',
        'heterogeneous'
    ]
    for query in query_terms:
        # Forks are okay. We use checksums to ensure uniqueness in
        # final dataset.
        #
        repos = g.search_repositories(query + ' fork:true sort:stars')

        for repo in repos:
            repo_modified = handle_repo(repo)

            # do nothing unless the repo is new or modified
            if not repo_modified:
                continue

            handle_file = partial(process_file, g, github_token, db, repo)

            # Iterate over the entire git tree of the repo's default
            # branch (usually 'master'). If a file ends with the .cl
            # extension, check to see if we already have it, else download
            # it.
            #
            try:
                branch = repo.default_branch
                tree_iterator = repo.get_git_tree(branch, recursive=True).tree
                for f in tree_iterator:
                    try:
                        handle_file(f)
                    except Exception:
                        errors_counter += 1
            except GithubException:
                # do nothing in case of error (such as an empty repo)
                pass

    print_counters()
    print("\n\ndone.")
    db.close()


_include_re = re.compile(r'\w*#include ["<](.*)[">]')
_parboil_re = re.compile(
    r'.+/benchmarks/parboil/benchmarks/(.+)/src/opencl_base/(.+\.cl)')


def get_path_id(path):
    match = re.match(_parboil_re, path)
    if match:
        return match.group(1) + '-' + match.group(2)
    else:
        return path


def inline_fs_headers(path, stack):
    stack.append(path)

    with open(path) as infile:
        src = infile.read()

    outlines = []
    for line in src.split('\n'):
        match = re.match(_include_re, line)
        if match:
            include_name = match.group(1)

            # try and resolve relative paths
            include_name = include_name.replace('../', '')

            include_path = os.path.join(os.path.dirname(path), include_name)

            if os.path.exists(include_path) and include_path not in stack:
                include_src = inline_headers(include_path, stack)
                outlines.append('// [FETCH] include: ' + include_path)
                outlines.append(include_src)
                outlines.append('// [FETCH] eof(' + include_path + ')')
            else:
                if include_path in stack:
                    outlines.append('// [FETCH] ignored recursive include: ' +
                                    include_path)
                else:
                    outlines.append('// [FETCH] 404 not found: ' +
                                    include_path)
        else:
            outlines.append(line)

    return '\n'.join(outlines)


def flatten(l):
    return [item for sublist in l for item in sublist]


def process_cl_file(db_path, path):
    """
    :param db_path: Path to output database.
    :param path: Path to input file.
    """
    db = dbutil.connect(db_path)
    c = db.cursor()

    log.debug("fetch {path}".format(path=fs.abspath(path)))
    try:
        contents = inline_fs_headers(path, [])
    except IOError:
        raise FetchError(
            "cannot read file '{path}'".format(path=fs.abspath(path)))
    id = get_path_id(path)
    c.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)',
              (id, contents))

    db.commit()
    c.close()


def content_db(db_path, in_db_path, table='PreprocessedFiles'):
    odb = dbutil.connect(db_path)
    idb = dbutil.connect(in_db_path)
    ic = idb.cursor()

    ic.execute('SELECT id,contents FROM {}'.format(table))
    rows = ic.fetchall()

    for id, contents in rows:
        kernels = clutil.get_cl_kernels(contents)
        ids = [clgen.checksum_str(kernel) for kernel in kernels]
        # print("{} kernels in {}".format(len(kernels), id))
        for kid, kernel in zip(ids, kernels):
            oc = odb.cursor()
            oc.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)',
                       (kid, kernel))
            odb.commit()


def fetch_fs(db_path, paths=[]):
    for path in paths:
        process_cl_file(db_path, path)


# Counters
kernel_counter = 0


def process_sample_file(db_path, sample_path, first_only=False,
                        max_kernel_len=5000):
    db = dbutil.connect(db_path)
    c = db.cursor()

    with open(sample_path) as infile:
        sample = infile.read()

    i = 0
    tail = 0
    offset = len('__kernel void ')
    while True:
        print('\r\033[Kkernel', i, end='')
        sys.stdout.flush()

        # Find the starting index of the next kernel.
        tail = sample.find('__kernel void ', tail)

        # If we didn't find another kernel, stop.
        if tail == -1:
            break

        # Find the end index of this kernel.
        head = clutil.get_cl_kernel_end_idx(sample, start_idx=tail,
                                            max_len=max_kernel_len)

        # Look for other ends
        end = sample.find('__kernel void ',
                          tail + offset, tail + offset + max_kernel_len)
        head = min(end, head) if end != -1 else head

        kernel = sample[tail:head]
        id = clgen.checksum_str(kernel)
        c.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)',
                  (id, kernel))
        tail = head
        i += 1
        if first_only:
            break
    print()
    db.commit()
    c.close()
    explore.explore(db_path)


def clgen_sample(db_path, samples_dir, sample_path, first_only):
    if samples_dir:
        files = [os.path.join(samples_dir, f) for f in os.listdir(samples_dir)
                 if os.path.isfile(os.path.join(samples_dir, f))]
        for sample_path in files:
            process_sample_file(db_path, sample_path, first_only=first_only)
    else:
        process_sample_file(db_path, sample_path, first_only=first_only)

    print("\r\033[K\ndone.")


# Counters
files_new_counter = 0
errors_counter = 0


class CLSmithException(clgen.CLgenError):
    pass


class HeaderNotFoundException(clgen.CLgenError):
    pass


def print_clsmith_counters():
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


def get_clsmith_program(db_path):
    global files_new_counter

    outputpath = 'CLProg.c'

    db = dbutil.connect(db_path)
    c = db.cursor()

    # TODO: CLSmith might not be in path
    cmd = ["CLSmith"]

    process = Popen(cmd)
    process.communicate()

    if process.returncode != 0:
        raise CLSmithException()

    with open(outputpath) as infile:
        contents = infile.read()

    contents = inline_headers(contents)

    sha = sha1(contents.encode('utf-8')).hexdigest()

    c.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)',
              (sha, contents))
    db.commit()
    db.close()
    files_new_counter += 1
    print_clsmith_counters()


def clsmith(db_path, target_num_kernels):
    global errors_counter

    print('generating', target_num_kernels, 'kernels to', db_path)

    db = dbutil.connect(db_path)
    c = db.cursor()
    c.execute('SELECT Count(*) FROM ContentFiles')
    num_kernels = c.fetchone()[0]
    while num_kernels < target_num_kernels:
        get_clsmith_program(db_path)
        c.execute('SELECT Count(*) FROM ContentFiles')
        num_kernels = c.fetchone()[0]

    print_counters()
    print("\n\ndone.")
    db.close()
