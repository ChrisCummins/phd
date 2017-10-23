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
from typing import List

import clgen
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


def _print_counters() -> None:
    """
    Print analytics counters.
    """
    print('\r\033[Kfiles: new ', files_new_counter,
          ', modified ', files_modified_counter,
          '. errors ', errors_counter,
          '. ', status_string[0:25],
          sep='', end='')
    sys.stdout.flush()


def _rate_limit(g) -> None:
    """
    Block on GitHub rate limit.

    Parameters
    ----------
    g
        GitHub connection.
    """
    global status_string
    remaining = g.get_rate_limit().rate.remaining
    while remaining < 100:
        sleep(1)
        status_string = 'WAITING ON RATE LIMIT'
        _print_counters()
        remaining = g.get_rate_limit().rate.remaining


def _process_repo(g, db, repo) -> bool:
    """
    GitHub repository handler.

    Determines if a repository needs to be scraped. There are two cases for
    this:
        * The repository has not already been visited.
        * The repository has been modified since it was last visited.

    Parameters
    ----------
    g
        GitHub connection.
    db : sqlite3.Connection
        Dataset.
    repo
        Repository.

    Returns
    -------
    bool
        True if repository should be scraped, else False.
    """
    global repos_new_counter
    global repos_modified_counter
    global repos_unchanged_counter
    global status_string

    _rate_limit(g)
    url = repo.url
    updated_at = str(repo.updated_at)
    name = repo.name
    status_string = name
    _print_counters()

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


_include_re = re.compile('\w*#include ["<](.*)[">]')
_sol_include_re = re.compile('\w*import ["<](\./)?(.*)[">];')


def _download_opencl_file(github_token: str, repo, url: str,
                          stack: List[str]) -> str:
    """
    Fetch file from GitHub.
    Recursively downloads and inlines headers.

    Parameters
    ----------
    github_token : str
        Authorization.
    repo
        Repository.
    url : str
        Path.
    stack : List[str]
        URL stack.

    Returns
    -------
    str
        File contents.
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
                include_src = _download_opencl_file(
                    github_token, repo, include_url)
                outlines.append(include_src)
            else:
                if not include_url:
                    outlines.append('// [FETCH] didnt find: ' + line)
                else:
                    outlines.append('// [FETCH] skipped: ' + line)
        else:
            outlines.append(line)

    return '\n'.join(outlines)


def _download_file(github_token: str, repo, url: str) -> str:
    """
    Fetch file from GitHub.

    Parameters
    ----------
    github_token : str
        Authorization.
    repo
        Repository.
    url : str
        Path.

    Returns
    -------
    str
        File contents.
    """
    response = json.loads(requests.get(
        url,
        headers={
            'Authorization': 'token ' + str(github_token)
        }
    ).content.decode('utf-8'))
    return b64decode(response['content']).decode('utf-8')


def _process_file(g, github_token: str, db, repo, file,
                  download_file_cb) -> bool:
    """
    GitHub file handler.

    Parameters
    ----------
    g
        GitHub connection.
    github_token : str
        Authorization.
    db : sqlite3.Connection
        Dataset.
    repo
        Repository.
    file
        File.

    Returns
    -------
    bool
        True on success, else False.
    """
    global files_new_counter
    global files_modified_counter
    global files_unchanged_counter
    global status_string

    url = file.url
    sha = file.sha
    path = file.path
    status_string = repo.name + '/' + path
    _print_counters()

    c = db.cursor()
    c.execute("SELECT sha FROM ContentMeta WHERE id=?", (url,))
    cached_sha = c.fetchone()

    # Do nothing unless checksums don't match
    if cached_sha and cached_sha[0] == sha:
        files_unchanged_counter += 1
        return False

    repo_url = repo.url
    contents = download_file_cb(github_token, repo, file.url)
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


def _scrape_github_for_files(db_path: str, github_username: str,
                             github_pw: str, github_token: str,
                             query_terms: List[str], file_is_intetesting,
                             download_file_cb):
    global errors_counter

    g = Github(github_username, github_pw)
    db = dbutil.connect(db_path)

    if not dbutil.is_github:
        raise clgen.UserError("not a GitHub database")

    # fetch the repositories to iterate over
    for query in query_terms:
        # forks are okay - we use checksums to ensure uniqueness in
        # final dataset
        repos = g.search_repositories(query + ' fork:true sort:stars')

        for repo in repos:
            # do nothing unless the repo is new or modified
            if not _process_repo(g, db, repo):
                continue

            # iterate over the entire git tree of the repo's default branch
            # (usually 'master'). If a file ends with the .cl extension, check
            # to see if we already have it, else download it
            try:
                branch = repo.default_branch
                tree_iterator = repo.get_git_tree(branch, recursive=True).tree
                for f in tree_iterator:
                    if file_is_intetesting(f):
                        try:
                            _process_file(g, github_token, db, repo, f,
                                          download_file_cb)
                        except Exception as e:
                            print(e)
                            sys.exit(1)
                            errors_counter += 1
            except GithubException:
                # do nothing in case of error (such as an empty repo)
                pass

    _print_counters()
    print("\n\ndone.")
    db.close()


def is_opencl_path(file) -> bool:
    """ We're only interested in OpenCL files. """
    return file.path.endswith('.cl') or file.path.endswith('.ocl')


def is_solidity_path(file) -> bool:
    """ We're only interesting in Solidity files. """
    return file.path.endswith('.sol')


def fetch_github(db_path: str, github_username: str, github_pw: str,
                 github_token: str, lang: str="opencl") -> None:
    """
    Download all of the Solidity on GitHub (!)

    Parameters
    ----------
    db_path : str
        Dataset path.
    github_username : str
        Authorization.
    github_pw : str
        Authorization.
    github_token : str
        Authorization.
    """
    if lang == "opencl":
        download_file_cb = _download_opencl_file
        file_is_intetesting = is_opencl_path
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
    elif lang == "solidity":
        download_file_cb = _download_file
        file_is_intetesting = is_solidity_path
        query_terms = [
            'solidity',
            'ethereum',
            'solc',
        ]
    else:
        raise ValueError(f"unsupported language '{lang}'")

    return _scrape_github_for_files(db_path, github_username, github_pw,
                                    github_token, query_terms,
                                    file_is_intetesting,
                                    download_file_cb)


def inline_fs_headers(path: str, stack: List[str]) -> str:
    """
    Recursively inline headers in file.

    Parameters
    ----------
    path : str
        File.
    stack : List[str]
        File stack.

    Returns
    -------
    str
        Inlined file.
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
                include_src = inline_fs_headers(include_path, stack)
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


def process_cl_file(db_path: str, path: str) -> None:
    """
    Process OpenCL file.

    Parameters
    ----------
    db_path : str
        Path to output database.
    path : str
        Path to input file.

    Raises
    ------
    FetchError
        In case of IO error.
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


def fetch(db_path: str, paths: List[str]=[]) -> None:
    """
    Fetch from a list of files.

    Parameters
    ----------
    db_path : str
        Output dataset.
    paths : List[str]
        List of file paths.
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
