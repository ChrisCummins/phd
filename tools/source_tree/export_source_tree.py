"""A script which exports the subset of this repository required for target(s).

This project is getting large. This has two major downsides:
  * Fresh checkouts of the git repository take longer and consume more space.
  * The large number of packages is confusing to newcomers.

I feel like there's a 90-10 rule that applies to this repo: 90% of people who
checkout this repo only need 10% of the code contained within it.
This script provides a way to export that 10%.
"""

import contextlib
import git
import github as github_lib
import pathlib
import tempfile
import typing

import getconfig
from datasets.github import api
from labm8 import app
from tools.source_tree import phd_workspace

FLAGS = app.FLAGS

app.DEFINE_list('targets', [], 'The bazel target(s) to export.')
app.DEFINE_list(
    'cp_files', [], 'A list of additional files to export. Each element in the '
    'list is either a relative path to export, or a mapping of '
    'relative paths in the form <src>:<dst>. E.g. '
    '`foo.py:bar/baz.txt` will export file `foo.py` to '
    'destination `bar/baz.txt`.')
app.DEFINE_list(
    'mv_files', [],
    'Each element in the list is a mapping of relative paths in the form '
    '<src>:<dst>. E.g. `foo.py:bar/baz.txt` will move file `foo.py` to '
    'destination `bar/baz.txt`.')
app.DEFINE_string('github_repo', None, 'Name of a GitHub repo to export to.')
app.DEFINE_boolean('github_create_repo', False,
                   'Whether to create the repo if it does not exist.')
app.DEFINE_boolean('github_repo_create_private', True,
                   'Whether to create new GitHub repos as private.')


def GetOrCreateRepoOrDie(github: github_lib.Github,
                         repo_name: str) -> github_lib.Repository:
  """Get the github repository to export to. Create it if it doesn't exist."""
  try:
    if FLAGS.github_create_repo:
      return api.GetOrCreateUserRepo(
          github,
          repo_name,
          description='PhD repo subtree export',
          homepage='https://github.com/ChrisCummins/phd',
          has_wiki=False,
          has_issues=False,
          private=FLAGS.github_repo_create_private)
    else:
      return api.GetUserRepo(github, repo_name)
  except api.RepoNotFoundError as e:
    app.FatalWithoutStackTrace(str(e))


@contextlib.contextmanager
def DestinationDirectoryFromFlags() -> pathlib.Path:
  """Get the export destination."""
  if FLAGS.github_repo:
    with tempfile.TemporaryDirectory(prefix='phd_tools_source_tree_') as d:
      yield pathlib.Path(d)
  else:
    yield pathlib.Path(FLAGS.destination)


def EXPORT(github_repo: str,
           targets: typing.List[str],
           copy_file_mapping: typing.Dict[str, str] = None,
           move_file_mapping: typing.Dict[str, str] = None):

  def _DoExport():
    source_path = pathlib.Path(getconfig.GetGlobalConfig().paths.repo_root)
    source_workspace = phd_workspace.PhdWorkspace(source_path)

    with tempfile.TemporaryDirectory(prefix='phd_tools_source_tree_') as d:
      destination = pathlib.Path(d)
      credentials = api.ReadGitHubCredentials(
          pathlib.Path('~/.githubrc').expanduser())
      connection = github_lib.Github(credentials.username, credentials.password)
      repo = GetOrCreateRepoOrDie(connection, github_repo)
      api.CloneRepoToDestination(repo, destination)
      destination_repo = git.Repo(destination)
      source_workspace.ExportToRepo(destination_repo, targets,
                                    copy_file_mapping, move_file_mapping)
      app.Log(1, 'Pushing changes to remote')
      destination_repo.git.push('origin')

  app.Run(_DoExport)


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  if not FLAGS.targets:
    raise app.UsageError('--targets must be one-or-more bazel targets')
  targets = list(sorted(set(FLAGS.targets)))

  def _GetFileMapping(f: str):
    if len(f.split(':')) == 2:
      return f.split(':')
    else:
      return f, f

  copy_file_tuples = [
      _GetFileMapping(f) for f in list(sorted(set(FLAGS.cp_files)))
  ]
  copy_file_mapping = {x[0]: x[1] for x in copy_file_tuples}

  move_file_tuples = [
      _GetFileMapping(f) for f in list(sorted(set(FLAGS.mv_files)))
  ]
  move_file_mapping = {x[0]: x[1] for x in move_file_tuples}

  with DestinationDirectoryFromFlags() as destination:
    connection = api.GetGithubConectionFromFlagsOrDie()
    repo = GetOrCreateRepoOrDie(connection, FLAGS.github_repo)
    api.CloneRepoToDestination(repo, destination)
    destination_repo = git.Repo(destination)
    source_workspace.ExportToRepo(destination_repo, targets, copy_file_mapping,
                                  move_file_mapping)
    destination_repo.git.push('origin')


if __name__ == '__main__':
  app.RunWithArgs(main)
