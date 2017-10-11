"""
Fetch C code on GitHub
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


def should_be_downloaded(path: str) -> bool:
    """
    Return whether file is opencl.

    Arguments:
        path (str): File.
    """
    return path.endswith('.c')

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

    if not should_be_downloaded(file.path):
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
        'language:C',
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
            # branch (usually 'master')
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

if __name__ == "__main__":
    import sys
    import os
    github(sys.argv[1], os.environ["GITHUB_USERNAME"],
           os.environ["GITHUB_PW"], os.environ["GITHUB_TOKEN"])
