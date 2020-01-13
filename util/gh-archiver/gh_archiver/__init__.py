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
import os
import shutil
import sys
from argparse import ArgumentParser, FileType, RawDescriptionHelpFormatter
from configparser import ConfigParser
from pathlib import Path
from typing import Iterator, List, TextIO, Tuple

import requests
from git import Repo
from github import Github, NamedUser
from github.Repository import Repository as GithubRepository


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
  parser.add_argument(
      "--gogs", action="store_true",
      help="Mirror repsoitories to gogs instance")
  parser.add_argument(
      "--gogs-uid", type=int, default=1,
      help="Gogs UID")
  parser.add_argument(
      "--gogsrc", metavar="<file>", type=FileType("r"),
      default=os.path.join(os.path.expanduser("~"), ".gogsrc"),
      help="Read Gogs server address and token from <file>")
  return parser


def get_github_config(githubrc: TextIO) -> Github:
  """ read GitHub config """
  config = ConfigParser()
  config.read(githubrc.name)

  try:
    github_username = config['User']['Username']
    github_pw = config['User']['Password']
  except KeyError as e:
    print(f'config variable {e} not set. Check ~/.githubrc',
          file=sys.stderr)
    sys.exit(1)

  return github_username, github_pw


def get_gogs_config(gogsrc: TextIO) -> Tuple[str, str]:
  """ read gogs configuration """
  config = ConfigParser()
  config.read(gogsrc.name)

  try:
    server = config['Server']['Address']
    token = config['User']['Token']
  except KeyError as e:
    print(f"config variable {e} not set. Check ~/.gogsrc",
          file=sys.stderr)
    sys.exit(1)

  return server, token


def get_repos(user, exclude_pattern: List[str]) -> Iterator[GithubRepository]:
  """ get user repositories """
  excluded = exclude_pattern.split(",")
  for repo in user.get_repos():
    if repo.name not in excluded:
      yield repo


def truncate(string: str, maxlen: int, suffix=" ...") -> str:
  """ truncate a string to a maximum length """
  if len(string) > maxlen:
    return string[:maxlen - len(suffix)] + suffix
  else:
    return string


def sanitize_description(string) -> str:
  """ make GitHub repo description compatible with gogs """
  if string:
    # the decode/encode dance. We need to get from ascii-encoded UTF-8, to
    # plain ascii, with unicode symbols stripped.
    return truncate(
        string \
          .encode('utf-8') \
          .decode('unicode_escape') \
          .encode('ascii', 'ignore') \
          .decode('ascii'), 255)
  else:
    return ''


def main():
  try:
    parser = get_argument_parser()
    args = parser.parse_args()
    github_username, github_pw = get_github_config(args.githubrc)
    g = Github(github_username, github_pw)

    if args.user == github_username:
      # AuthenticatedUser provides access to private repos
      user: AuthenticatedUser = g.get_user()
    else:
      user: NamedUser = g.get_user(username)

    repos = list(get_repos(user, args.excludes))

    if args.gogs:
      gogs_server, gogs_token = get_gogs_config(args.gogsrc)

    outdir = Path(args.outdir, parents=True, exists_ok=True)

    # delete any files which are not GitHub repos first, if necessary
    if args.delete:
      if args.gogs:
        repo_names = [r.name.lower() + ".git" for r in repos]
      else:
        repo_names = [r.name for r in repos]

      for path in outdir.iterdir():
        local_repo_name = os.path.basename(path)

        if local_repo_name not in repo_names:
          print(f"removing {local_repo_name}")
          if path.is_dir():
            shutil.rmtree(path)
          else:
            os.remove(path)

    for repo in repos:
      if args.gogs:
        local_path = outdir / Path(repo.name.lower() + ".git")
      else:
        local_path = outdir / repo.name

      # remove any file of the same name
      if local_path.exists() and not local_path.is_dir():
        os.remove(local_path)

      if args.gogs:
        # Mirror to gogs instance
        if not local_path.is_dir():
          sys.stdout.write(f"mirroring {repo.name} ... ")
          headers = {
            "Authorization": f"token {gogs_token}",
          }
          data = {
            "auth_username": github_username,
            "auth_password": github_pw,
            "repo_name": truncate(repo.name, 255),
            "clone_addr": repo.clone_url,
            "uid": args.gogs_uid,
            "description": sanitize_description(repo.description),
            "private": False,
            "mirror": True,
          }

          def pretty(d):
            import json
            print(json.dumps(d, sort_keys=True,
                             indent=4, separators=(',', ': ')))

          r = requests.post(gogs_server + "/api/v1/repos/migrate",
                            headers=headers, data=data)
          print(r.status_code)
          if r.status_code < 200 or r.status_code >= 300:
            pretty(headers)
            pretty(data)
            print()
            print("status", r.status_code)
            pretty(r.json())
            print(len(data["description"]))
            sys.exit(1)
      else:
        # Local clone
        if local_path.is_dir():
          print(f"updating {repo.name}")
        else:
          print(f"cloning {repo.name}:{repo.default_branch}")
          Repo.clone_from(repo.git_url, local_path,
                          branch=repo.default_branch)

        local_repo = Repo(local_path)
        for remote in local_repo.remotes:
          remote.fetch()
        for submodule in local_repo.submodules:
          submodule.update(init=True)
  except KeyboardInterrupt:
    print()
    sys.exit(1)
