#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
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
from labm8 import crypto
from labm8 import fs
from subprocess import Popen
from time import sleep

import clgen
from clgen import clutil
from clgen import dbutil
from clgen import log


class FetchError(clgen.CLgenError):
    """
    Module error.
    """
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


def print_repo_details(repo) -> None:
    """
    Print GitHub repo details.

    Arguments:
        repo: Repository.
    """
    print('url:', repo.url)
    print('owner:', repo.owner.email)
    print('name:', repo.name)
    print('fork:', repo.fork)
    print('stars:', repo.stargazers_count)
    print('contributors:', len([x for x in repo.get_contributors()]))
    print('forks:', repo.forks)
    print('created_at:', repo.created_at)
    print('updated_at:', repo.updated_at)


def print_file_details(file) -> None:
    """
    Print GitHub file details.

    Arguments:
        file: File.
    """
    print('url:', file.url)
    print('path:', file.path)
    print('sha:', file.sha)
    print('size:', file.size)


def print_counters() -> None:
    """
    Print analytics counters.
    """
    print('\r\033[Kfiles: new ', files_new_counter,
          ', modified ', files_modified_counter,
          '. errors ', errors_counter,
          '. ', status_string[0:25],
          sep='', end='')
    sys.stdout.flush()


def rate_limit(g) -> None:
    """
    Block on GitHub rate limit.

    Arguments:
        g: GitHub connection.
    """
    global status_string
    remaining = g.get_rate_limit().rate.remaining
    while remaining < 100:
        sleep(1)
        status_string = 'WAITING ON RATE LIMIT'
        print_counters()
        remaining = g.get_rate_limit().rate.remaining


def process_repo(g, db, repo) -> bool:
    """
    GitHub repository handler.

    Determines if a repository needs to be scraped. There are two cases for
    this:
        * The repository has not already been visited.
        * The repository has been modified since it was last visited.

    Arguments:
        g: GitHub connection.
        db (sqlite3.Connection): Dataset.
        repo: Repository.

    Returns:
        bool: True if repository should be scraped, else False.
    """
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


def is_opencl_path(path: str) -> bool:
    """
    Return whether file is opencl.

    Arguments:
        path (str): File.
    """
    return path.endswith('.cl') or path.endswith('.ocl')

_include_re = re.compile('\w*#include ["<](.*)[">]')


def download_file(github_token: str, repo, url: str, stack: list) -> str:
    """
    Fetch file from GitHub.

    Recursively downloads and inlines headers.

    Arguments:
        github_token (str): Authorization.
        repo: Repository.
        url (str): Path.
        stack (str[]): URL stack.

    Returns:
        str: File contents.
    """
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


def process_file(g, github_token: str, db, repo, file) -> bool:
    """
    GitHub file handler.

    Arguments:
        g: GitHub connection.
        github_token (str): Authorization.
        db (sqlite3.Connection): Dataset.
        repo: Repository.
        file: File.

    Return:
        bool: True on success, else False.
    """
    global files_new_counter
    global files_modified_counter
    global files_unchanged_counter
    global status_string

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


def github(db_path: str, github_username: str, github_pw: str,
           github_token: str) -> None:
    """
    Download all of the OpenCL on GitHub (!)

    Shortcomings of this appraoch:
        * Only includes exclusively OpenCL files, no inline strings.
        * Occasionally (< 1%) can't find headers to include.

    Arguments:
        db_path (str): Dataset path.
        github_username (str): Authorization.
        github_pw (str): Authorization.
        github_token (str): Authorization.
    """
    global errors_counter

    g = Github(github_username, github_pw)
    db = dbutil.connect(db_path)

    if not dbutil.is_github:
        raise clgen.UserError("not a GitHub database")

    handle_repo = partial(process_repo, g, db)

    # fetch the repositories to iterate over. Since opencl isn't
    # treated as a first-class language by GitHub, we can't use the
    # 'language=' keyword for queries, so instead we through a much
    # wider net and filter the results afterwards.
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
        # forks are okay - we use checksums to ensure uniqueness in
        # final dataset
        repos = g.search_repositories(query + ' fork:true sort:stars')

        for repo in repos:
            repo_modified = handle_repo(repo)

            # do nothing unless the repo is new or modified
            if not repo_modified:
                continue

            handle_file = partial(process_file, g, github_token, db, repo)

            # iterate over the entire git tree of the repo's default
            # branch (usually 'master'). If a file ends with the .cl
            # extension, check to see if we already have it, else download
            # it
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


def inline_fs_headers(path: str, stack: list) -> str:
    """
    Recursively inline headers in file.

    Arguments:
        path (str): File.
        stack (str[]): File stack.

    Returns:
        str: Inlined file.
    """
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


def flatten(l: list) -> list:
    """
    Flattens a list of lists.

    Arguments:
        l (list of list): Input.

    Returns:
        list: Flattened list.
    """
    return [item for sublist in l for item in sublist]


def process_cl_file(db_path: str, path: str) -> None:
    """
    Process OpenCL file.

    Arguments:
        db_path (str): Path to output database.
        path (str): Path to input file.

    Raises:
        FetchError: In case of IO error.
    """
    db = dbutil.connect(db_path)
    c = db.cursor()

    log.debug("fetch {path}".format(path=fs.abspath(path)))
    try:
        contents = inline_fs_headers(path, [])
    except IOError:
        raise FetchError(
            "cannot read file '{path}'".format(path=fs.abspath(path)))
    c.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)',
              (path, contents))

    db.commit()
    c.close()


def content_db(db_path: str, in_db_path: str,
               table: str='PreprocessedFiles') -> None:
    """
    Fetch kernels from a content database.

    Arguments:
        db_path (str): Output path.
        in_db_path (str): Input path.
        table (str, optional): Table to fetch from.
    """
    odb = dbutil.connect(db_path)
    idb = dbutil.connect(in_db_path)
    ic = idb.cursor()

    ic.execute('SELECT id,contents FROM {}'.format(table))
    rows = ic.fetchall()

    for id, contents in rows:
        kernels = clutil.get_cl_kernels(contents)
        ids = [crypto.sha1_str(kernel) for kernel in kernels]
        # print("{} kernels in {}".format(len(kernels), id))
        for kid, kernel in zip(ids, kernels):
            oc = odb.cursor()
            oc.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)',
                       (kid, kernel))
            odb.commit()


def fetch_fs(db_path: str, paths: list=[]) -> None:
    """
    Fetch from a list of files.

    Arguments:
        db_path (str): Output dataset.
        paths (str[]): List of file paths.
    """
    paths = fs.files_from_list(*paths)  # expand directories

    db = dbutil.connect(db_path)
    c = db.cursor()

    for path in paths:
        log.debug("fetch", path)
        try:
            contents = inline_fs_headers(path, [])
        except IOError:
            db.commit()
            raise FetchError(
                "cannot read file '{path}'".format(path=fs.abspath(path)))
        c.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)',
                  (path, contents))

    db.commit()


# Counters
files_new_counter = 0
errors_counter = 0


class CLSmithException(clgen.CLgenError):
    """
    CLSmith error.
    """
    pass


class HeaderNotFoundException(clgen.CLgenError):
    """
    Unable to locate header file.
    """
    pass


def print_clsmith_counters() -> None:
    """
    Print CLSmith counters.
    """
    print('\r\033[Kfiles: new ', files_new_counter,
          '. errors ', errors_counter,
          sep='', end='')
    sys.stdout.flush()


def include_clsmith_path(name: str, header_paths: list) -> str:
    """
    Fetch path to CLSmith header.

    Arguments:
        name (str): Header name.
        header_paths (str[]): Directories containing CLSmith headers.

    Returns:
        str: Header path.
    """
    for dir in header_paths:
        path = os.path.join(os.path.expanduser(dir), name)
        if os.path.exists(path):
            return path
    raise HeaderNotFoundException(name)


def inline_clsmith_headers(src: str, header_paths: list) -> str:
    """
    Inline CLSmith headers.

    Arguments:
        str (str): CLSmith source.
        header_paths (str[]): Directories containing CLSmith headers.

    Returns:
        str: CLSmith source with headers inlined.
    """
    outlines = []
    for line in src.split('\n'):
        match = re.match(_include_re, line)
        if match:
            include_name = match.group(1)

            path = include_clsmith_path(include_name, header_paths)
            with open(path) as infile:
                header = infile.read()
                outlines.append(inline_clsmith_headers(header))
        else:
            outlines.append(line)

    return '\n'.join(outlines)


def get_clsmith_program(db_path: str,
                        header_paths: list=[
                            "~/clsmith/runtime", "~/clsmith/build"]) -> None:
    """
    Generate a program using CLSmith and add to dataset.

    Arguments:
        db_path (str): Path to output dataset.
        header_paths (str[]): Directories containing CLSmith headers.
    """
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

    contents = inline_clsmith_headers(contents, header_paths)

    sha = sha1(contents.encode('utf-8')).hexdigest()

    c.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)',
              (sha, contents))
    db.commit()
    db.close()
    files_new_counter += 1
    print_clsmith_counters()


def clsmith(db_path: str, target_num_kernels: int) -> None:
    """
    Generate kernels using CLSmith.

    Arguments:
        db_path (str): Path to dataset.
        target_num_kernels (int): Number of kernels to generate.
    """
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
