"""Unit tests for //tools/source_tree."""

import pathlib
import pytest
import git
from labm8 import test
from labm8 import fs

from tools.source_tree import source_tree

FLAGS = test.FLAGS


def test_TemporaryGitRemote_remote_is_available(repo_with_history: git.Repo,
                                                empty_repo: git.Repo):
  temporary_remote = source_tree.TemporaryGitRemote(
      empty_repo, repo_with_history.working_tree_dir)
  with temporary_remote as temporary_remote_name:
    assert empty_repo.remote(temporary_remote_name)


def test_TemporaryGitRemote_remote_fetch(repo_with_history: git.Repo,
                                         empty_repo: git.Repo):
  temporary_remote = source_tree.TemporaryGitRemote(
      empty_repo, repo_with_history.working_tree_dir)
  with temporary_remote as temporary_remote_name:
    empty_repo.remote(temporary_remote_name).fetch()


def test_TemporaryGitRemote_remote_is_removed_after_scope(
    repo_with_history: git.Repo, empty_repo: git.Repo):
  temporary_remote = source_tree.TemporaryGitRemote(
      empty_repo, repo_with_history.working_tree_dir)
  with temporary_remote as temporary_remote_name:
    pass
  assert temporary_remote_name not in empty_repo.remotes


def test_GetCommitsInOrder_count(repo_with_history: git.Repo):
  commits = source_tree.GetCommitsInOrder(repo_with_history)
  assert len(commits) == 3


def test_GetCommitsInOrder_empty_repo_count(empty_repo: git.Repo):
  commits = source_tree.GetCommitsInOrder(empty_repo)
  assert len(commits) == 0


def test_GetCommitsInOrder_head_is_last(repo_with_history: git.Repo):
  commits = source_tree.GetCommitsInOrder(repo_with_history)
  head = repo_with_history.commit('HEAD')
  assert commits[-1] == head


def test_GetCommitsInOrder_tail_is_first(repo_with_history: git.Repo):
  commits = source_tree.GetCommitsInOrder(repo_with_history)
  head_offset = len(commits) - 1
  tail = repo_with_history.commit(f'HEAD~{head_offset}')
  assert commits[0] == tail


def test_GetCommitsInOrder_head_ref(repo_with_history: git.Repo):
  all_commits = source_tree.GetCommitsInOrder(repo_with_history)
  all_except_one_commits = source_tree.GetCommitsInOrder(repo_with_history,
                                                         head_ref='HEAD~')
  assert len(all_commits) - 1 == len(all_except_one_commits)


def test_GetCommitsInOrder_tail_ref(repo_with_history: git.Repo):
  commits = source_tree.GetCommitsInOrder(repo_with_history, tail_ref='HEAD~')
  assert len(commits) == 1


def test_MaybeExportCommitSubset_interesting_no_interesting_file(
    empty_repo: git.Repo, empty_repo2: git.Repo):
  src, dst = empty_repo, empty_repo2

  # Create a README and commit it.
  readme = fs.Write(
      pathlib.Path(src.working_tree_dir) / 'README.txt',
      "Hello, world!\n".encode('utf-8'))
  src.index.add([str(readme)])
  src_commit = src.index.commit("First commit, add a README")

  # Make commit available to dest.
  dst.git.remote('add', 'origin', src.working_tree_dir)
  dst.remote('origin').fetch()

  dst_commit = source_tree.MaybeExportCommitSubset(src_commit, dst, [])
  assert not dst_commit


def test_MaybeExportCommitSubset_interesting_file_added(empty_repo: git.Repo,
                                                        empty_repo2: git.Repo):
  src, dst = empty_repo, empty_repo2

  # Create a README and commit it.
  readme = fs.Write(
      pathlib.Path(src.working_tree_dir) / 'README.txt',
      "Hello, world!\n".encode('utf-8'))
  src.index.add([str(readme)])
  author = git.Actor(name='Joe Bloggs', email='joe@bloggs.com')
  committer = git.Actor(name='Eve', email='eve@msn.com')
  src_commit = src.index.commit("First commit, add a README",
                                author=author,
                                committer=committer)

  # Make commit available to dest.
  dst.git.remote('add', 'origin', src.working_tree_dir)
  dst.remote('origin').fetch()

  dst_commit = source_tree.MaybeExportCommitSubset(src_commit, dst,
                                                   ['README.txt'])
  assert dst_commit

  # Test that commit message has export annotation.
  assert dst_commit.message == (
      f'First commit, add a README\n[Exported from {src_commit.hexsha}]')
  # Test that commit attributes are propagated.
  assert dst_commit.author == author
  assert dst_commit.committer == committer
  # File is in dst working tree.
  assert (pathlib.Path(dst.working_tree_dir) / 'README.txt').is_file()
  assert fs.Read(
      pathlib.Path(dst.working_tree_dir) / 'README.txt') == ('Hello, world!\n')


def test_MaybeExportCommitSubset_interesting_file_subset(
    empty_repo: git.Repo, empty_repo2: git.Repo):
  """Test that only interesting files are exported."""
  src, dst = empty_repo, empty_repo2

  # Create a README and LICENSE and commit it.
  readme = fs.Write(
      pathlib.Path(src.working_tree_dir) / 'README.txt',
      "Hello, world!\n".encode('utf-8'))
  license_ = fs.Write(
      pathlib.Path(src.working_tree_dir) / 'LICENSE.md',
      "Hello, license!\n".encode('utf-8'))
  src.index.add([str(readme), str(license_)])
  src_commit = src.index.commit("First commit, add a README and LICENSE")

  # Make commit available to dest.
  dst.git.remote('add', 'origin', src.working_tree_dir)
  dst.remote('origin').fetch()

  dst_commit = source_tree.MaybeExportCommitSubset(src_commit, dst,
                                                   ['README.txt'])
  assert dst_commit

  # Test that the right files are in dst working tree.
  assert (pathlib.Path(dst.working_tree_dir) / 'README.txt').is_file()
  assert not (pathlib.Path(dst.working_tree_dir) / 'LICENSE.md').is_file()


def test_MaybeExportCommitSubset_uninteresting_file_renamed(
    empty_repo: git.Repo, empty_repo2: git.Repo):
  """Test the case where an unintersting file is renamed to an interesting
  file.

  My initial implementation failed this test, raising an
  git.exc.UnmergedEntriesError when attempting to commit the new file.
  """
  src, dst = empty_repo, empty_repo2

  # Create a README and commit it.
  readme = fs.Write(
      pathlib.Path(src.working_tree_dir) / 'README.txt',
      "Hello, world!\n".encode('utf-8'))
  src.index.add([str(readme)])
  src.index.commit("First commit, add a README")

  src.index.move(['README.txt', 'README.md'])
  src_commit = src.index.commit("Rename README")

  # Make commit available to dest.
  dst.git.remote('add', 'origin', src.working_tree_dir)
  dst.remote('origin').fetch()

  dst_commit = source_tree.MaybeExportCommitSubset(src_commit, dst,
                                                   ['README.md'])
  assert dst_commit

  # Test that the right files exist.
  assert not (pathlib.Path(dst.working_tree_dir) / 'README.txt').is_file()
  assert (pathlib.Path(dst.working_tree_dir) / 'README.md').is_file()


def test_MaybeExportCommitSubset_commit_conflict(empty_repo: git.Repo,
                                                 empty_repo2: git.Repo):
  """Test that partial commit conflict doesn't lose history."""
  src, dst = empty_repo, empty_repo2

  # Create a README commit it.
  readme = fs.Write(
      pathlib.Path(src.working_tree_dir) / 'README.txt',
      "Hello, world!\n".encode('utf-8'))
  src.index.add([str(readme)])
  src.index.commit("First commit, add a README")

  # Add a LICENSE and modify the README.
  readme = fs.Write(
      pathlib.Path(src.working_tree_dir) / 'README.txt',
      "Goodbye, world!\n".encode('utf-8'))
  license_ = fs.Write(
      pathlib.Path(src.working_tree_dir) / 'LICENSE.md',
      "Hello?\n".encode('utf-8'))
  src.index.add([str(readme), str(license_)])
  src_commit = src.index.commit("Second commit, add LICENSE and change README")

  # Make commits available to dest.
  dst.git.remote('add', 'origin', src.working_tree_dir)
  dst.remote('origin').fetch()

  dst_commit = source_tree.MaybeExportCommitSubset(src_commit, dst,
                                                   ['LICENSE.md'])
  assert dst_commit

  # Test that the right files are in dst working tree.
  assert not (pathlib.Path(dst.working_tree_dir) / 'README.txt').is_file()
  assert (pathlib.Path(dst.working_tree_dir) / 'LICENSE.md').is_file()


def test_ExportCommitsThatTouchFiles_integration_test_readme(
    repo_with_history: git.Repo, empty_repo: git.Repo):
  src, dst = repo_with_history, empty_repo

  src_commits = source_tree.GetCommitsInOrder(src)

  # Make commits available to dest.
  dst.git.remote('add', 'origin', src.working_tree_dir)
  dst.remote('origin').fetch()

  source_tree.ExportCommitsThatTouchFiles(src_commits, dst, ['README.txt'])

  dst_commits = source_tree.GetCommitsInOrder(dst)
  assert len(dst_commits) == 1
  assert (pathlib.Path(dst.working_tree_dir) / "README.txt").is_file()
  assert not (pathlib.Path(dst.working_tree_dir) / 'src' / "main.c").is_file()
  assert not (pathlib.Path(dst.working_tree_dir) / 'src' / "Makefile").is_file()


def test_ExportCommitsThatTouchFiles_integration_test_readme_and_makefile(
    repo_with_history: git.Repo, empty_repo: git.Repo):
  src, dst = repo_with_history, empty_repo

  src_commits = source_tree.GetCommitsInOrder(src)

  # Make commits available to dest.
  dst.git.remote('add', 'origin', src.working_tree_dir)
  dst.remote('origin').fetch()

  source_tree.ExportCommitsThatTouchFiles(src_commits, dst,
                                          ['README.txt', 'src/Makefile'])

  dst_commits = source_tree.GetCommitsInOrder(dst)
  assert (pathlib.Path(dst.working_tree_dir) / "README.txt").is_file()
  assert not (pathlib.Path(dst.working_tree_dir) / 'src' / "main.c").is_file()
  assert not (pathlib.Path(dst.working_tree_dir) / 'src' / "Makefile").is_file()
  assert len(dst_commits) == 3


def test_ExportCommitsThatTouchFiles_integration_test_nothing_interesting(
    repo_with_history: git.Repo, empty_repo: git.Repo):
  src, dst = repo_with_history, empty_repo

  src_commits = source_tree.GetCommitsInOrder(src)

  # Make commits available to dest.
  dst.git.remote('add', 'origin', src.working_tree_dir)
  dst.remote('origin').fetch()

  source_tree.ExportCommitsThatTouchFiles(src_commits, dst, ['foo'])

  dst_commits = source_tree.GetCommitsInOrder(dst)
  assert len(dst_commits) == 0
  assert not (pathlib.Path(dst.working_tree_dir) / "README.txt").is_file()
  assert not (pathlib.Path(dst.working_tree_dir) / 'src' / "main.c").is_file()
  assert not (pathlib.Path(dst.working_tree_dir) / 'src' / "Makefile").is_file()


def test_ExportGitHistoryForFiles_duplicate_exports(repo_with_history: git.Repo,
                                                    empty_repo: git.Repo):
  src, dst = repo_with_history, empty_repo

  source_tree.ExportGitHistoryForFiles(
      src, dst, ['README.txt', 'src/main.c', 'src/Makefile'])
  commits = source_tree.GetCommitsInOrder(dst)

  # Do it again. Since we've already exported everything, there should be
  # nothing more.
  source_tree.ExportGitHistoryForFiles(
      src, dst, ['README.txt', 'src/main.c', 'src/Makefile'])
  commits2 = source_tree.GetCommitsInOrder(dst)

  assert len(commits) == len(commits2)


def test_ExportGitHistoryForFiles_split_export(repo_with_history: git.Repo,
                                               empty_repo: git.Repo):
  src, dst = repo_with_history, empty_repo

  source_tree.ExportGitHistoryForFiles(
      src, dst, ['README.txt', 'src/main.c', 'src/Makefile'], head_ref='HEAD~')
  commits = source_tree.GetCommitsInOrder(dst)

  # Do it again. Since we've already exported everything, there should be
  # nothing more.
  source_tree.ExportGitHistoryForFiles(
      src, dst, ['README.txt', 'src/main.c', 'src/Makefile'])
  commits2 = source_tree.GetCommitsInOrder(dst)

  assert len(commits) + 1 == len(commits2)


def test_MaybeGetHexShaOfLastExportedCommit_empty_repo(empty_repo: git.Repo):
  assert not source_tree.MaybeGetHexShaOfLastExportedCommit(empty_repo)


def test_MaybeGetHexShaOfLastExportedCommit_empty_repo(empty_repo: git.Repo):
  assert not source_tree.MaybeGetHexShaOfLastExportedCommit(empty_repo)


def test_MaybeGetHexShaOfLastExportedCommit_file(repo_with_history: git.Repo,
                                                 empty_repo: git.Repo):
  src, dst = repo_with_history, empty_repo

  source_tree.ExportGitHistoryForFiles(
      src, dst, ['README.txt', 'src/main.c', 'src/Makefile'])
  commits = source_tree.GetCommitsInOrder(src)

  assert (
      source_tree.MaybeGetHexShaOfLastExportedCommit(dst) == commits[-1].hexsha)


if __name__ == '__main__':
  test.Main()
