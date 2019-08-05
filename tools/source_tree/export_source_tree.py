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
import sys
import typing

import getconfig
from datasets.github import api
from labm8 import app
from tools.source_tree import phd_workspace

FLAGS = app.FLAGS

app.DEFINE_list('targets', [], 'The bazel target(s) to export.')
app.DEFINE_list('excluded_targets', [],
                'A list of bazel targets to exclude from export.')
app.DEFINE_list(
    'extra_files', [], 'A list of additional files to export. Each element in '
    'the list is a relative path to export. E.g. `bar/baz.txt`.')
app.DEFINE_list(
    'move_file_mapping', [],
    'Each element in the list is a mapping of relative paths in the form '
    '<src>:<dst>. E.g. `foo.py:bar/baz.txt` will move file `foo.py` to '
    'destination `bar/baz.txt`.')
app.DEFINE_string('github_repo', None, 'Name of a GitHub repo to export to.')
app.DEFINE_boolean('github_create_repo', False,
                   'Whether to create the repo if it does not exist.')
app.DEFINE_boolean('github_repo_create_private', True,
                   'Whether to create new GitHub repos as private.')
app.DEFINE_boolean('export_source_tree_print_files', False,
                   'Print the files that will be exported and terminate.')
app.DEFINE_boolean(
    'ignore_last_export', False,
    'If true, run through the entire git history. Otherwise, '
    'continue from the last commit exported. Use this flag if '
    'the set of exported files changes.')


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
  except (api.RepoNotFoundError, OSError) as e:
    app.FatalWithoutStackTrace(str(e))


def EXPORT(github_repo: str,
           targets: typing.List[str],
           excluded_targets: typing.List[str] = None,
           extra_files: typing.List[str] = None,
           move_file_mapping: typing.Dict[str, str] = None,
           resume_export: bool = True,
           run_handler=app.Run) -> None:
  """Custom entry-point to export source-tree.

  This should be called from a bare python script, before flags parsing.

  Args:
    github_repo: The name of the GitHub repo to export to.
    targets: A list of bazel targets to export. These targets, and their
      dependencies, will be exported. These arguments are passed unmodified to
      bazel query, so `/...` and `:all` labels are expanded, e.g.
      `//some/package/to/export/...`. All targets should be absolute, and
      prefixed with '//'.
    extra_files: A list of additional files to export.
    move_file_mapping: A dictionary of <src,dst> relative paths listing files
      which should be moved from their respective source location to the
      destination.
  """
  excluded_targets = excluded_targets or []
  extra_files = extra_files or []
  move_file_mapping = move_file_mapping or {}

  def _DoExport():
    source_path = pathlib.Path(getconfig.GetGlobalConfig().paths.repo_root)
    source_workspace = phd_workspace.PhdWorkspace(source_path)

    with tempfile.TemporaryDirectory(prefix=f'phd_export_{github_repo}_') as d:
      destination = pathlib.Path(d)
      credentials = api.ReadGitHubCredentials(
          pathlib.Path('~/.githubrc').expanduser())
      connection = github_lib.Github(credentials.username, credentials.password)
      repo = GetOrCreateRepoOrDie(connection, github_repo)
      api.CloneRepoToDestination(repo, destination)
      destination_repo = git.Repo(destination)

      src_files = source_workspace.GetAllSourceTreeFiles(
          targets, excluded_targets, extra_files, move_file_mapping)
      if FLAGS.export_source_tree_print_files:
        print('\n'.join(src_files))
        sys.exit(0)

      source_workspace.ExportToRepo(destination_repo, targets, src_files,
                                    extra_files, move_file_mapping,
                                    resume_export)
      app.Log(1, 'Pushing changes to remote')
      destination_repo.git.push('origin')

  run_handler(_DoExport)


def main():
  if not FLAGS.targets:
    raise app.UsageError('--targets must be one-or-more bazel targets')
  targets = list(sorted(set(FLAGS.targets)))

  excluded_targets = set(FLAGS.excluded_targets)

  def _GetFileMapping(f: str):
    if len(f.split(':')) == 2:
      return f.split(':')
    else:
      return f, f

  extra_files = list(sorted(set(FLAGS.extra_files)))

  move_file_tuples = [
      _GetFileMapping(f) for f in list(sorted(set(FLAGS.move_file_mapping)))
  ]
  move_file_mapping = {x[0]: x[1] for x in move_file_tuples}

  EXPORT(github_repo=FLAGS.github_repo,
         targets=targets,
         excluded_targets=excluded_targets,
         extra_files=extra_files,
         move_file_mapping=move_file_mapping,
         resume_export=not FLAGS.ignore_last_export,
         run_handler=lambda x: x())


if __name__ == '__main__':
  app.Run(main)
