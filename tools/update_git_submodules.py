#!/usr/bin/env python3
#
# A script to initialize all repository submodules.
#
# Usage:
#
#     $ python ./tools/update_git_submodules
#
# Must be run from repository root.
#
import os
import re
import subprocess


def GetGitRemote(path):
  """Get the git remote for this repo."""
  remotes = subprocess.check_output(
    ["git", "-C", str(path), "remote", "-v"]
  ).decode("utf-8")
  return remotes.split("\n")[0].split()[1]


def RewriteGitSubmodulesToHttps(path):
  """Rewrite git@ prefixed submodules to https://."""
  gitmodules = os.path.join(path, ".gitmodules")
  with open(gitmodules) as f:
    modules = f.read()
  new_modules = re.sub(r"(url\s*=\s*)git@([^:]+):", r"\1https://\2/", modules)
  assert "git@" not in new_modules
  with open(gitmodules, "w") as f:
    f.write(new_modules)


if __name__ == "__main__":
  phd_root = os.environ.get("PHD", os.getcwd())

  # Git submodules are checked in using SSH protocol URLs, but if the user
  # cloned this repo using the HTTP protocol then there's a good chance they
  # will want to use that for the submodules too. To support this, we
  # optionally rewrite git@ to https:// URLs in submodules.
  if os.path.isdir(os.path.join(phd_root, ".git")):
    git_remote = GetGitRemote(phd_root)
    print("Git remote:", git_remote)
    gitmodules_file = os.path.join(phd_root, ".gitmodules")
    if os.path.isfile(gitmodules_file) and git_remote.startswith("https://"):
      print("Rewriting .gitmodules to use https://")
      RewriteGitSubmodulesToHttps(phd_root)

  print("Initializing and updating git submodules")
  subprocess.check_call(
    ["git", "-C", phd_root, "submodule", "update", "--init", "--recursive"]
  )
