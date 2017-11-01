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
import dateutil
import dateutil.parser
import editdistance
import json
import os
import re
import requests
import sqlite3
import subprocess
import sys

from base64 import b64decode
from functools import partial
from github import Github, GithubException
from hashlib import sha1
from io import open
from labm8 import crypto
from labm8 import fs
from pathlib import Path
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
        match = re.match(include_re, line)
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


def fetch_repos(db_path: Path, indir: Path, lang: clgen.Language) -> None:
    db = dbutil.connect(db_path)

    if not dbutil.is_github(db):
        raise clgen.UserError("not a GitHub database")

    c = db.cursor()

    for directory in fs.ls(indir, abspaths=True):
        # hacky hardcoded interpretation of `git remote -v`
        gitdir = fs.path(directory, ".git")
        output = subprocess.check_output(
            ["git", "--git-dir", gitdir, "remote", "-v"],
            universal_newlines=True)
        url = output.split("\n")[0].split("\t")[1].split(" ")[0]
        name = fs.basename(directory)

        output = subprocess.check_output(
            f"git --git-dir {gitdir} rev-list --format=format:'%ai' " +
            f"--max-count=1 $(git --git-dir {gitdir} rev-parse HEAD) | tail -n1",
            shell=True, universal_newlines=True)
        try:
            updated_at = dateutil.parser.parse(output)
        except ValueError:
            log.error(f"failed to process {name} {url}")
            continue

        c.execute("SELECT updated_at FROM Repositories WHERE url=?", (url,))
        cached_updated_at = c.fetchone()

        # Do nothing unless updated timestamps don't match
        # if cached_updated_at and cached_updated_at[0] >= updated_at:
        #     log.verbose(name, "already in database")
        #     continue

        c.execute("DELETE FROM Repositories WHERE url=?", (url,))
        c.execute("INSERT INTO Repositories VALUES(?,?,?,?,?,?,?,?,?)",
              (url, "<unknown>", name, 0, 0, 0, 0, updated_at, updated_at))

        name_str = " -o ".join([f"-name '*{ext}'" for ext in clgen.file_extensions(lang)])
        output = subprocess.check_output(
            f"find {directory} -type f {name_str} | grep -v '.git/' || true",
            shell=True, universal_newlines=True)
        files = [x.strip() for x in output.split("\n") if x.strip()]

        # nothing to import
        if not len(files):
            # log.verbose("no files in", name)
            continue

        log.verbose("processing", len(files), "files in", name)
        for path in files:
            relpath = path[len(directory) + 1:]
            try:
                contents = inline_fs_headers(path, [], lang=lang)
                sha = crypto.sha1_str(contents)
                c.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)',
                          (sha, contents))
                c.execute("INSERT OR IGNORE INTO ContentMeta VALUES(?,?,?,?,?)",
                          (sha, relpath, url, sha, len(contents)))
            except UnicodeDecodeError:
                log.warning("non UTF-8 file", path)

        db.commit()
        c = db.cursor()

def fetch_github(db_path: str, github_username: str, github_pw: str,
                 github_token: str, lang: clgen.Language=clgen.Language.OPENCL) -> None:
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
    if lang == clgen.Language.OPENCL:
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
    elif lang == clgen.Language.SOLIDITY:
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


def inline_fs_headers(path: Path, stack: List[str],
                      lang: clgen.Language=clgen.Language.OPENCL,
                      topdir: Path=None) -> str:
    """
    Recursively inline headers in file.

    Parameters
    ----------
    path : str
        File.
    stack : List[str]
        File stack.
    topdir : Path
        The top level directory to stop searching for includes in.

    Returns
    -------
    str
        Inlined file.
    """
    stack.append(path)

    if topdir is None:
        topdir = fs.dirname(path)
    # shell escaped top directory
    escp_topdir = topdir.replace('"', '\\"')

    include_re = clgen.include_regexp(lang)

    with open(path, encoding="utf-8") as infile:
        src = infile.read()

    outlines = []
    for line in src.split('\n'):
        match = re.match(include_re, line)
        if match:
            # We have an import to inline!
            include = match.group("path")

            # Search for files with that name in the repository
            include_basename = fs.basename(include)
            esc_basename = include_basename.replace('"', '\\"')
            candidates = [x for x in
                subprocess.check_output(
                    f'find "{escp_topdir}" -type f -name {esc_basename}',
                    shell=True, universal_newlines=True)\
                    .split('\n')
                if x]

            # Select which file to inline:
            if len(candidates) == 1:
                # If there's exactly one match, then we're done:
                file_to_inline = candidates[0]
            elif len(candidates) > 1:
                # We have multiple candidates to inline, so we'll compare the
                # full paths (relative to the top directory) to select the one
                # whose name is the closest match:
                rel_matches = [match[len(topdir) + 1:] for match in candidates]
                distances = [editdistance.eval(include, path) for path in rel_matches]
                min_distance = min(distances)
                file_to_inline = candidates[distances.index(min_distance)]
                log.debug(f"Inferred include '{file_to_inline}' from '{line}' with distance {min_distance}")
            else:
                # We didn't find anything suitable:
                file_to_inline = None

            # Process the inline file:
            if file_to_inline in stack:
                # We've already inlined this file, so ignore it:
                outlines.append(clgen.format_as_comment(
                    lang, f'[FETCH] ignored_include({line})'))
            elif file_to_inline:
                # Inline the file by recursively expanding its contents:
                outlines.append(clgen.format_as_comment(
                    lang, f'[FETCH] begin_include({line})'))
                inline_src = inline_fs_headers(file_to_inline, stack)
                outlines.append(inline_src)
                outlines.append(clgen.format_as_comment(
                    lang, f'[FETCH] end_include({line})'))
            else:
                # We didn't find anything suitable, so keep the original
                # include:
                outlines.append(clgen.format_as_comment(
                    lang, f'[FETCH] not_found({line})'))
                outlines.append(line)
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
