# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
Clone and update a GitHub user's repos locally.
"""
import calendar
import os
import shutil
import sys
import time
import os

from argparse import ArgumentParser, FileType, RawDescriptionHelpFormatter
from configparser import ConfigParser
from datetime import datetime
from git import Repo
from github import Github, GithubException, NamedUser
from github.Repository import Repository as GithubRepository
from pathlib import Path
from typing import Iterator, TextIO

__copyright__ = "Copyright (C) 2017 Chris Cummins <chrisc.101@gmail.com>."

def get_argument_parser() -> ArgumentParser:
    """ construct the argument parser """
    parser = ArgumentParser(
        description=__doc__,
        epilog=__copyright__,
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument(
        "user", metavar="<username>",
        help="GitHub username")
    parser.add_argument(
        "-o", "--outdir", metavar="<dir>", default=".",
        help="Write output to <dir> (default: current directory)")
    parser.add_argument(
        "--delete", action="store_true",
        help="Delete all other files in output directory")
    parser.add_argument(
        "--exclude", dest="excludes", metavar="<repo1,repo2 ...>", default="",
        help="Comma-separated list of repository names to exclude")
    parser.add_argument(
        "--githubrc", metavar="<file>", type=FileType("r"),
        default=os.path.join(os.path.expanduser("~"), ".githubrc"),
        help="Read GitHub username and password from <file>")
    return parser


def get_github_connection(githubrc: TextIO) -> Github:
    """ connect to GitHub """
    config = ConfigParser()
    config.read(githubrc.name)

    try:
        github_username = config['User']['Username']
        github_pw = config['User']['Password']
    except KeyError as e:
        print(f'config variable {e} not set. Check {args.githubrc}',
              file=sys.stderr)
        sys.exit(1)

    return Github(github_username, github_pw)


def get_repos(g: Github, username: str,
              exclude_pattern: str) -> Iterator[GithubRepository]:
    """ get user repositories """
    user: NamedUser = g.get_user(username)
    excluded = exclude_pattern.split(",")
    for repo in user.get_repos():
        if repo.name not in excluded:
            yield repo


def main():
    try:
        parser = get_argument_parser()
        args = parser.parse_args()
        g = get_github_connection(args.githubrc)

        outdir = Path(args.outdir, parents=True, exists_ok=True)
        repos = list(get_repos(g, args.user, args.excludes))

        # delete any files which are not GitHub repos first, if necessary
        if args.delete:
            repo_names = [r.name for r in repos]
            for path in outdir.iterdir():
                basename = os.path.basename(path)
                if basename not in repo_names:
                    print(f"removing {basename}")
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        os.remove(path)

        for repo in repos:
            local_path = outdir / repo.name

            # remove any file of the same name
            if local_path.exists() and not local_path.is_dir():
                os.remove(local_path)

            if local_path.is_dir():
                print(f"updating {repo.name}")
            else:
                print(f"cloning {repo.name}")
                Repo.clone_from(repo.git_url, local_path)

            local_repo = Repo(local_path)
            for remote in local_repo.remotes:
                remote.fetch()
            for submodule in local_repo.submodules:
                submodule.update(init=True)
    except KeyboardInterrupt:
        print()
        sys.exit(1)
