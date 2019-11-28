"""Library for manipulating source trees."""
import contextlib
import re
import time
import typing

import git

from labm8.py import app
from labm8.py import humanize

FLAGS = app.FLAGS


@contextlib.contextmanager
def TemporaryGitRemote(repo: git.Repo, remote_url: str) -> str:
  """Add a git remote and return it's name.

  Args:
    repo: The repo to add the remote to.
    remote_url: The URL of the remote to add.

  Returns:
    The name of the remote.
  """
  remote_name = f'tmp_import_{int(time.time() * 1e6)}'
  repo.git.remote('add', remote_name, remote_url)
  yield remote_name
  repo.git.remote('remove', remote_name)


def GetCommitsInOrder(
    repo: git.Repo,
    head_ref: str = "HEAD",
    tail_ref: typing.Optional[str] = None) -> typing.List[git.Commit]:
  """Get a list of all commits, in chronological order from old to new.

  Args:
    repo: The repo to list the commits of.
    head_ref: The starting point for iteration, e.g. the commit closest to
      head.
    tail_ref: The end point for iteration, e.g. the commit closest to tail.
      This commit is NOT included in the returned values.

  Returns:
    A list of git.Commit objects.
  """

  def TailCommitIterator():
    stop_commit = repo.commit(tail_ref)
    for commit in repo.iter_commits(head_ref):
      if commit == stop_commit:
        break
      yield commit

  if tail_ref:
    app.Log(1, 'Resuming export from commit `%s`', tail_ref)
    commit_iter = TailCommitIterator()
  else:
    app.Log(1, 'Exporting entire git history')
    commit_iter = repo.iter_commits(head_ref)

  try:
    return list(reversed(list(commit_iter)))
  except git.GitCommandError:
    # If HEAD is not found, an exception is raised.
    return []


def MaybeExportCommitSubset(
    commit: git.Commit, repo: git.Repo,
    files_of_interest: typing.Set[str]) -> typing.Optional[git.Commit]:
  """Filter the parts of the given commit that touch the files_of_interest and
  commit them. If the commit doesn't touch anything interesting, nothing is
  commited.

  Args:
    repo: The git repo to add the commit to.

  Returns:
    A git commit, if one is created, else None.
  """
  try:
    # Apply the diff of the commit to be exported to the repo.
    repo.git.cherry_pick('--no-commit', '--allow-empty', commit)
    unmerged_to_add = set()
  except git.GitCommandError:
    # If cherry pick fails its because of merge conflicts.
    unmerged_paths = set(
        [path for path, _ in repo.index.unmerged_blobs().items()])
    unmerged_to_add = set(
        [path for path in unmerged_paths if path in files_of_interest])
    unmerged_to_rm = unmerged_paths - unmerged_to_add
    if unmerged_to_add:
      app.Log(2, 'Adding %s unmerged files', len(unmerged_to_add))
      # We have to remove an unmerged file before adding it again, else
      # the commit will fail with unmerged error.
      repo.index.remove(list(unmerged_to_add))
      repo.index.add(list(unmerged_to_add))
    if unmerged_to_rm:
      app.Log(2, 'Removing %s unmerged files', len(unmerged_to_rm))
      repo.index.remove(list(unmerged_to_rm))

  # Filter the changed files and exclude those that aren't interesting.
  modified_paths = set([path for (path, _), _ in repo.index.entries.items()])
  paths_to_unstage = set(
      [path for path in modified_paths if path not in files_of_interest])
  paths_to_commit = modified_paths - paths_to_unstage
  if paths_to_unstage:
    app.Log(2, 'Removing %s uninteresting files', len(paths_to_unstage))
    repo.index.remove(list(paths_to_unstage))

  if not repo.is_dirty():
    app.Log(2, 'Skipping empty commit')
    repo.git.clean('-xfd')
    return

  # I'm not sure about this one. The idea here is that in cases where there is
  # a merge error, checkout the file directly from the commit that the merge
  # error came from.
  try:
    modified_paths = [
        path.a_path for path in repo.index.diff('HEAD').iter_change_type('M')
    ]
    modified_unmerged_paths = set(modified_paths).union(unmerged_to_add)
    if modified_unmerged_paths:
      repo.git.checkout(commit, *modified_unmerged_paths)
  except (git.BadName, git.GitCommandError):
    pass

  # Append the hexsha of the original commit that this was exported from. This
  # can be used to determine the starting point for incremental exports, by
  # reading the commit messages and parsing this statement.
  message = f'{commit.message}\n[Exported from {commit}]'
  app.Log(2, 'Committing %s files', len(paths_to_commit))

  def _FormatPythonDatetime(dt):
    """Make python datetime compatabile with commit date args."""
    return dt.replace(tzinfo=None).replace(microsecond=0).isoformat()

  new_commit = repo.index.commit(
      message=message,
      author=commit.author,
      committer=commit.committer,
      author_date=_FormatPythonDatetime(commit.authored_datetime),
      commit_date=_FormatPythonDatetime(commit.committed_datetime),
      skip_hooks=True)
  repo.git.reset('--hard')
  repo.git.clean('-xfd')
  return new_commit


def MaybeGetHexShaOfLastExportedCommit(
    repo: git.Repo, head_ref: str = "HEAD") -> typing.List[str]:
  """The the SHA1 of the most recently exported commit.

  Args:
    repo: The repo to iterate over.
    head_ref: The starting point for iteration, e.g. the commit closest to
      head.

  Returns:
    The hex SHA1 of the last exported commited, else None.
  """
  export_re = re.compile(r'\n\[Exported from ([a-fA-F0-9]{40})\]')
  try:
    for commit in repo.iter_commits(head_ref):
      if '\n[Exported from ' in commit.message:
        match = export_re.search(commit.message)
        assert match
        return match.group(1)
  except git.GitCommandError:
    # Raise if no HEAD, i.e. no commits.
    pass
  return None


def ExportCommitsThatTouchFiles(commits_in_order: typing.List[git.Commit],
                                destiantion: git.Repo,
                                files_of_interest: typing.Set[str]) -> int:
  """Filter and apply the commits that touch the given files of interest.

  The commits are applied in the order provided.
  """
  exported_commit_count = 0
  total_commit_count = humanize.Commas(len(commits_in_order))
  for i, commit in enumerate(commits_in_order):
    app.Log(1, 'Processing commit %s of %s (%.2f%%) %s', humanize.Commas(i + 1),
            total_commit_count, ((i + 1) / len(commits_in_order)) * 100, commit)
    if MaybeExportCommitSubset(commit, destiantion, files_of_interest):
      exported_commit_count += 1
  return exported_commit_count


def ExportGitHistoryForFiles(source: git.Repo,
                             destination: git.Repo,
                             files_of_interest: typing.Set[str],
                             head_ref: str = 'HEAD',
                             resume_export: bool = True) -> int:
  """Apply the parts of the git history from the given source repo """
  if destination.is_dirty():
    raise OSError("Repo `{destination.working_tree_dir}` is dirty")

  with TemporaryGitRemote(destination, source.working_tree_dir) as remote:
    destination.remote(remote).fetch()
    tail = None
    if resume_export:
      tail = MaybeGetHexShaOfLastExportedCommit(destination)
    commits_in_order = GetCommitsInOrder(source,
                                         head_ref=head_ref,
                                         tail_ref=tail)
    if not commits_in_order:
      app.Log(1, 'Nothing to export!')
      return 0
    app.Log(1, 'Exporting history from %s commits',
            humanize.Commas(len(commits_in_order)))
    return ExportCommitsThatTouchFiles(commits_in_order, destination,
                                       files_of_interest)
