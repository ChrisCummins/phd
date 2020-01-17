# Copyright 2017-2020 Chris Cummins <chrisc.101@gmail.com>
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
"""Mirror a Github user's repos locally.

This program fetches a Github user's repositories and mirrors them to a local
directory. New repositories are cloned, existing repositories are fetched from
remote.

Setup:

    Create a Github personal access token by visiting
    <https://github.com/settings/tokens>. If you intend to mirror your own
    private repositories, you should enable private repository permissions when
    creating the token. Else, no permissions are required.

    Create a ~/.githubrc file containing your Github username and the personal
    access token you just created:

        $ cat <<EOF > ~/.githubrc
        [User]
        Username = YourUsername

        [Tokens]
        gh_archiver = YourAccessToken
        EOF

    You should restrict multi-user access to this file:

        $ chmod 0600 ~/.githubrc

Usage:

    Mirror a Github user's repositories to a directory using:

        $ gh_archiver --user <github_username> --outdir <path>
"""
import os
import pathlib
import shutil
import sys
from configparser import ConfigParser
from typing import Iterator
from typing import Tuple

import requests
from git import Repo
from github import AuthenticatedUser
from github import Github
from github import NamedUser
from github.Repository import Repository as GithubRepository

from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_string("user", None, "Github username")
app.DEFINE_output_path(
  "outdir",
  pathlib.Path("~/gh_archiver").expanduser(),
  "The directory to clone repositories to.",
)
app.DEFINE_boolean(
  "delete",
  True,
  "Delete all other files in the output directory. Disabling this feature will "
  "cause deleted repositories to accumulate.",
)
app.DEFINE_list(
  "repo",
  [],
  "White list repostories to include. If set, only these repositories will be "
  "cloned.",
)
app.DEFINE_list("exclude", [], "A list of repositories to exclude.")
app.DEFINE_input_path(
  "githubrc",
  pathlib.Path("~/.githubrc").expanduser(),
  "Path to a file containing Github credentials See --help.",
)
app.DEFINE_boolean("gogs", False, "Mirror repositories to gogs instance.")
app.DEFINE_integer("gogs_uid", 1, "The gogs UID.")
app.DEFINE_output_path(
  "gogsrc",
  pathlib.Path("~/.gogsrc").expanduser(),
  "File to read Gogs server address and token from.",
)
app.DEFINE_boolean(
  "force_https",
  False,
  "If set, force using HTTPS git remotes, even when SSH is available.",
)


def get_github_config() -> Github:
  """ read Github config """
  config = ConfigParser()
  config.read(FLAGS.githubrc)

  try:
    github_username = config["User"]["Username"]
    github_token = config["Tokens"]["gh_archiver"]
  except KeyError as e:
    print(f"config variable {e} not set in: {FLAGS.githubrc}", file=sys.stderr)
    sys.exit(1)

  return github_username, github_token


def get_gogs_config() -> Tuple[str, str]:
  """ read gogs configuration """
  if not FLAGS.gogsrc.is_file():
    raise app.UsageError(f"--gogsrc file not found: {FLAGS.gogsrc}")

  config = ConfigParser()
  config.read(FLAGS.gogsrc)

  try:
    server = config["Server"]["Address"]
    token = config["User"]["Token"]
  except KeyError as e:
    print(f"config variable {e} not set. Check {FLAGS.gogsrc}", file=sys.stderr)
    sys.exit(1)

  return server, token


def get_repos(user) -> Iterator[GithubRepository]:
  """ get user repositories """
  excluded = set(FLAGS.exclude)
  included = set(FLAGS.repo)
  for repo in user.get_repos():
    if included and repo.name not in included:
      continue
    if repo.name not in excluded:
      yield repo


def truncate(string: str, maxlen: int, suffix=" ...") -> str:
  """ truncate a string to a maximum length """
  if len(string) > maxlen:
    return string[: maxlen - len(suffix)] + suffix
  else:
    return string


def sanitize_description(string) -> str:
  """ make Github repo description compatible with gogs """
  if string:
    # the decode/encode dance. We need to get from ascii-encoded UTF-8, to
    # plain ascii, with unicode symbols stripped.
    return truncate(
      string.encode("utf-8")
      .decode("unicode_escape")
      .encode("ascii", "ignore")
      .decode("ascii"),
      255,
    )
  else:
    return ""


def main():
  if not FLAGS.user:
    raise app.UsageError("--user not set")

  try:
    github_username, github_token = get_github_config()
    g = Github(github_token)

    if FLAGS.user == github_username:
      # AuthenticatedUser provides access to private repos
      user: AuthenticatedUser = g.get_user()
    else:
      user: NamedUser = g.get_user(FLAGS.user)

    repos = list(get_repos(user))

    if FLAGS.gogs:
      gogs_server, gogs_token = get_gogs_config()

    FLAGS.outdir.mkdir(exist_ok=True, parents=True)

    # delete any files which are not Github repos first, if necessary
    if FLAGS.delete:
      if FLAGS.gogs:
        repo_names = [r.name.lower() + ".git" for r in repos]
      else:
        repo_names = [r.name for r in repos]

      for path in FLAGS.outdir.iterdir():
        local_repo_name = os.path.basename(path)

        if local_repo_name not in repo_names:
          print(f"removing {local_repo_name}")
          if path.is_dir():
            shutil.rmtree(path)
          else:
            os.remove(path)

    errors = False
    for repo in repos:
      if FLAGS.gogs:
        local_path = FLAGS.outdir / pathlib.Path(repo.name.lower() + ".git")
      else:
        local_path = FLAGS.outdir / repo.name

      # remove any file of the same name
      if local_path.exists() and not local_path.is_dir():
        os.remove(local_path)

      if FLAGS.gogs:
        # Mirror to gogs instance
        if not local_path.is_dir():
          sys.stdout.write(f"mirroring {repo.name} ... ")
          headers = {
            "Authorization": f"token {gogs_token}",
          }
          data = {
            "auth_username": github_username,
            "auth_token": github_token,
            "repo_name": truncate(repo.name, 255),
            "clone_addr": repo.clone_url,
            "uid": FLAGS.gogs_uid,
            "description": sanitize_description(repo.description),
            "private": False,
            "mirror": True,
          }

          def pretty(d):
            import json

            print(
              json.dumps(d, sort_keys=True, indent=4, separators=(",", ": "))
            )

          r = requests.post(
            gogs_server + "/api/v1/repos/migrate", headers=headers, data=data
          )
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
          try:
            Repo.clone_from(
              repo.clone_url if FLAGS.force_https else repo.git_url,
              local_path,
              branch=repo.default_branch,
            )
          except Exception as e:
            errors = True
            print(f"ERROR: {local_path} {type(e).__name__}", file=sys.stderr)

        try:
          local_repo = Repo(local_path)
          for remote in local_repo.remotes:
            remote.fetch()
          for submodule in local_repo.submodules:
            submodule.update(init=True)
        except Exception as e:
          errors = True
          print(f"ERROR: {local_path} {type(e).__name__}", file=sys.stderr)

    return sys.exit(1 if errors else 0)
  except KeyboardInterrupt:
    print()
    sys.exit(1)


if __name__ == "__main__":
  app.Run(main)
