"""Clone and update a GitHub user's repos locally."""
import os
import pathlib
import shutil
import sys
import typing
from configparser import ConfigParser

import github
import requests
from absl import app
from absl import flags
from absl import logging
from git import Repo
from github.Repository import Repository as GithubRepository

from datasets.github import non_hemetic_credentials_file
from labm8 import jsonutil


FLAGS = flags.FLAGS

flags.DEFINE_string('user', None, 'GitHub username.')
flags.DEFINE_bool(
    'delete', False,
    'Delete all other files in output directory except the mirrored repos.')
flags.DEFINE_string('dst', None, 'Destination directory for mirrored repos.')
flags.DEFINE_list('excludes', [], 'List of repository names to exclude.')

# Gogs.
flags.DEFINE_bool('gogs', False, 'Mirror repositories to gogs instance.')
flags.DEFINE_integer('gogs_uid', 1, 'Gogs UID.')
flags.DEFINE_string(
    'gogsrc', '~/.gogsrc', "Read Gogs server address and token from path.")


def GogsCredentialsFromFileOrDie(path: pathlib.Path) -> typing.Tuple[str, str]:
  """Read gogs configuration."""
  config = ConfigParser()
  config.read(path)

  try:
    server = config['Server']['Address']
    token = config['User']['Token']
  except KeyError as e:
    logging.fatal("config variable %s not set. Check ~/.gogsrc", e)

  return server, token


def GetUsersRepos(
    user, excluded: typing.List[str]) -> typing.Iterator[GithubRepository]:
  """ get user repositories """
  for repo in user.get_repos():
    if repo.name not in excluded:
      yield repo


def Truncate(string: str, maxlen: int, suffix=" ...") -> str:
  """Truncate a string to a maximum length."""
  if len(string) > maxlen:
    return string[:maxlen - len(suffix)] + suffix
  else:
    return string


def SanitizeRepoDescription(string: str) -> str:
  """Make GitHub repo description compatible with gogs."""
  if string:
    # The decode/encode dance. We need to get from ascii-encoded UTF-8 to
    # plain ascii with unicode symbols stripped.
    return Truncate(
        string \
          .encode('utf-8') \
          .decode('unicode_escape') \
          .encode('ascii', 'ignore') \
          .decode('ascii'), 255)
  else:
    return ''


def GetGitHubConnectionAndUserFromFlags():
  github_credentials = non_hemetic_credentials_file.GitHubCredentialsFromFlag()
  g = non_hemetic_credentials_file.GetGitHubConnection(github_credentials)

  # Get the user who's repos we're mirroring.
  if FLAGS.user == github_credentials.username:
    # AuthenticatedUser provides access to private repos.
    user: github.AuthenticatedUser = g.get_user()
  else:
    user: github.NamedUser = g.get_user(FLAGS.username)

  return github_credentials, g, user


def main(argv) -> None:
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  try:
    github_credentials, g, user = GetGitHubConnectionAndUserFromFlags()

    repos = list(GetUsersRepos(user, FLAGS.excludes))

    if FLAGS.gogs:
      gogs_server, gogs_token = GogsCredentialsFromFileOrDie(
          pathlib.Path(FLAGS.gogsrc).expanduser())
      logging.info('Using gogs server %s', gogs_server)
      gogs_headers = {
        "Authorization": f"token {gogs_token}",
      }
      # Make a quick API call so that we fail now if connection does not
      # succeed.
      # TODO(cec): Neater error handling.
      r = requests.get(gogs_server + "/api/v1/repos/search", data={
        "limit": 1,
      })
      assert r.status_code == 200

    # Create the destination directory.
    if not FLAGS.dst:
      raise app.UsageError("--dst must be set")
    outdir = pathlib.Path(FLAGS.dst)
    outdir.mkdir(parents=True, exist_ok=True)

    # Delete any files which are not GitHub repos first, if necessary
    if FLAGS.delete:
      if FLAGS.gogs:
        repo_names = [r.name.lower() + ".git" for r in repos]
      else:
        repo_names = [r.name for r in repos]

      for path in outdir.iterdir():
        local_repo_name = os.path.basename(path)

        if local_repo_name not in repo_names:
          logging.info(f"removing {local_repo_name}")
          if path.is_dir():
            shutil.rmtree(path)
          else:
            os.remove(path)

    for repo in repos:
      if FLAGS.gogs:
        local_path = outdir / pathlib.Path(repo.name.lower() + ".git")
      else:
        local_path = outdir / repo.name

      # remove any file of the same name
      if local_path.exists() and not local_path.is_dir():
        os.remove(local_path)

      if FLAGS.gogs:
        # Mirror to gogs instance
        if local_path.is_dir():
          logging.info("Already mirrored %s", repo.name)
        else:
          logging.info("Mirroring %s from %s", repo.name, repo.clone_url)
          data = {
            "auth_username": github_credentials.username,
            "auth_password": github_credentials.password,
            "repo_name": Truncate(repo.name, 255),
            "clone_addr": repo.clone_url,
            "uid": FLAGS.gogs_uid,
            "description": SanitizeRepoDescription(repo.description),
            "private": False,
            "mirror": True,
          }

          r = requests.post(gogs_server + "/api/v1/repos/migrate",
                            headers=gogs_headers, data=data)
          logging.info("Mirrored %s: %s", repo.name, r.status_code)
          if r.status_code < 200 or r.status_code >= 300:
            logging.error("Headers: %s", jsonutil.format_json(headers))
            logging.error("Data: %s", jsonutil.format_json(data))
            logging.error("Status: %s", r.status_code)
            logging.error("Return: %s", jsonutil.format_json(r.json()))
            logging.fatal("Repository %s failed to clone with status %d",
                          repo.name, r.status_code)
      else:
        # Local clone
        if local_path.is_dir():
          logging.info("Updating %s", repo.name)
        else:
          clone_url = repo.ssh_url
          logging.info("Cloning %s:%s from %s", repo.name, repo.default_branch,
                       clone_url)
          Repo.clone_from(clone_url, local_path, branch=repo.default_branch)

        local_repo = Repo(local_path)
        for remote in local_repo.remotes:
          remote.fetch()
        for submodule in local_repo.submodules:
          submodule.update(init=True)
  except KeyboardInterrupt:
    logging.info('interrupt')
    sys.exit(1)


if __name__ == '__main__':
  app.run(main)
