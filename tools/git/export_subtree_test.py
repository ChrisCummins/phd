"""Unit tests for //tools/git:export_subtree."""
import pathlib

import git

from labm8.py import fs
from labm8.py import test
from tools.git import export_subtree

pytest_plugins = ["tools.git.test.fixtures"]

FLAGS = test.FLAGS


def test_GetCommitsInOrder_count(repo_with_history: git.Repo):
  commits = export_subtree.GetCommitsInOrder(repo_with_history)
  assert len(commits) == 3


def test_GetCommitsInOrder_empty_repo_count(empty_repo: git.Repo):
  commits = export_subtree.GetCommitsInOrder(empty_repo)
  assert len(commits) == 0


def test_GetCommitsInOrder_head_is_last(repo_with_history: git.Repo):
  commits = export_subtree.GetCommitsInOrder(repo_with_history)
  head = repo_with_history.commit("HEAD")
  assert commits[-1] == head


def test_GetCommitsInOrder_tail_is_first(repo_with_history: git.Repo):
  commits = export_subtree.GetCommitsInOrder(repo_with_history)
  head_offset = len(commits) - 1
  tail = repo_with_history.commit(f"HEAD~{head_offset}")
  assert commits[0] == tail


def test_GetCommitsInOrder_head_ref(repo_with_history: git.Repo):
  all_commits = export_subtree.GetCommitsInOrder(repo_with_history)
  all_except_one_commits = export_subtree.GetCommitsInOrder(
    repo_with_history, head_ref="HEAD~"
  )
  assert len(all_commits) - 1 == len(all_except_one_commits)


def test_GetCommitsInOrder_tail_ref(repo_with_history: git.Repo):
  commits = export_subtree.GetCommitsInOrder(
    repo_with_history, tail_ref="HEAD~"
  )
  assert len(commits) == 1


def test_MaybeExportCommitSubset_interesting_no_interesting_file(
  empty_repo: git.Repo, empty_repo2: git.Repo
):
  src, dst = empty_repo, empty_repo2

  # Create a README and commit it.
  readme = fs.Write(
    pathlib.Path(src.working_tree_dir) / "README.txt",
    "Hello, world!\n".encode("utf-8"),
  )
  src.index.add([str(readme)])
  src_commit = src.index.commit("First commit, add a README")

  # Make commit available to dest.
  dst.git.remote("add", "origin", src.working_tree_dir)
  dst.remote("origin").fetch()

  dst_commit = export_subtree.MaybeExportCommitSubset(src_commit, dst, [])
  assert not dst_commit


def test_MaybeExportCommitSubset_interesting_file_added(
  empty_repo: git.Repo, empty_repo2: git.Repo
):
  src, dst = empty_repo, empty_repo2

  # Create a README and commit it.
  readme = fs.Write(
    pathlib.Path(src.working_tree_dir) / "README.txt",
    "Hello, world!\n".encode("utf-8"),
  )
  src.index.add([str(readme)])
  author = git.Actor(name="Joe Bloggs", email="joe@bloggs.com")
  committer = git.Actor(name="Eve", email="eve@msn.com")
  src_commit = src.index.commit(
    "First commit, add a README", author=author, committer=committer
  )

  # Make commit available to dest.
  dst.git.remote("add", "origin", src.working_tree_dir)
  dst.remote("origin").fetch()

  dst_commit = export_subtree.MaybeExportCommitSubset(
    src_commit, dst, ["README.txt"]
  )
  assert dst_commit

  # Test that commit message has export annotation.
  assert dst_commit.message == (
    f"First commit, add a README\n[Exported from {src_commit.hexsha}]"
  )
  # Test that commit attributes are propagated.
  assert dst_commit.author == author
  assert dst_commit.committer == committer
  # File is in dst working tree.
  assert (pathlib.Path(dst.working_tree_dir) / "README.txt").is_file()
  assert fs.Read(pathlib.Path(dst.working_tree_dir) / "README.txt") == (
    "Hello, world!\n"
  )


def test_MaybeExportCommitSubset_interesting_file_subset(
  empty_repo: git.Repo, empty_repo2: git.Repo
):
  """Test that only interesting files are exported."""
  src, dst = empty_repo, empty_repo2

  # Create a README and LICENSE and commit it.
  readme = fs.Write(
    pathlib.Path(src.working_tree_dir) / "README.txt",
    "Hello, world!\n".encode("utf-8"),
  )
  license_ = fs.Write(
    pathlib.Path(src.working_tree_dir) / "LICENSE.md",
    "Hello, license!\n".encode("utf-8"),
  )
  src.index.add([str(readme), str(license_)])
  src_commit = src.index.commit("First commit, add a README and LICENSE")

  # Make commit available to dest.
  dst.git.remote("add", "origin", src.working_tree_dir)
  dst.remote("origin").fetch()

  dst_commit = export_subtree.MaybeExportCommitSubset(
    src_commit, dst, ["README.txt"]
  )
  assert dst_commit

  # Test that the right files are in dst working tree.
  assert (pathlib.Path(dst.working_tree_dir) / "README.txt").is_file()
  assert not (pathlib.Path(dst.working_tree_dir) / "LICENSE.md").is_file()


def test_MaybeExportCommitSubset_uninteresting_file_renamed(
  empty_repo: git.Repo, empty_repo2: git.Repo
):
  """Test the case where an unintersting file is renamed to an interesting
  file.

  My initial implementation failed this test, raising an
  git.exc.UnmergedEntriesError when attempting to commit the new file.
  """
  src, dst = empty_repo, empty_repo2

  # Create a README and commit it.
  readme = fs.Write(
    pathlib.Path(src.working_tree_dir) / "README.txt",
    "Hello, world!\n".encode("utf-8"),
  )
  src.index.add([str(readme)])
  src.index.commit("First commit, add a README")

  src.index.move(["README.txt", "README.md"])
  src_commit = src.index.commit("Rename README")

  # Make commit available to dest.
  dst.git.remote("add", "origin", src.working_tree_dir)
  dst.remote("origin").fetch()

  dst_commit = export_subtree.MaybeExportCommitSubset(
    src_commit, dst, ["README.md"]
  )
  assert dst_commit

  # Test that the right files exist.
  assert not (pathlib.Path(dst.working_tree_dir) / "README.txt").is_file()
  assert (pathlib.Path(dst.working_tree_dir) / "README.md").is_file()


def test_MaybeExportCommitSubset_commit_conflict(
  empty_repo: git.Repo, empty_repo2: git.Repo
):
  """Test that partial commit conflict doesn't lose history."""
  src, dst = empty_repo, empty_repo2

  # Create a README commit it.
  readme = fs.Write(
    pathlib.Path(src.working_tree_dir) / "README.txt",
    "Hello, world!\n".encode("utf-8"),
  )
  src.index.add([str(readme)])
  src.index.commit("First commit, add a README")

  # Add a LICENSE and modify the README.
  readme = fs.Write(
    pathlib.Path(src.working_tree_dir) / "README.txt",
    "Goodbye, world!\n".encode("utf-8"),
  )
  license_ = fs.Write(
    pathlib.Path(src.working_tree_dir) / "LICENSE.md",
    "Hello?\n".encode("utf-8"),
  )
  src.index.add([str(readme), str(license_)])
  src_commit = src.index.commit("Second commit, add LICENSE and change README")

  # Make commits available to dest.
  dst.git.remote("add", "origin", src.working_tree_dir)
  dst.remote("origin").fetch()

  dst_commit = export_subtree.MaybeExportCommitSubset(
    src_commit, dst, {"LICENSE.md"}
  )
  assert dst_commit

  # Test that the right files are in dst working tree.
  assert not (pathlib.Path(dst.working_tree_dir) / "README.txt").is_file()
  assert (pathlib.Path(dst.working_tree_dir) / "LICENSE.md").is_file()


def test_ExportSubtree_integration_test_readme(
  repo_with_history: git.Repo, empty_repo: git.Repo
):
  src, dst = repo_with_history, empty_repo

  export_subtree.ExportSubtree(src, dst, {"README.txt"})

  dst_commits = export_subtree.GetCommitsInOrder(dst)
  assert len(dst_commits) == 1
  assert (pathlib.Path(dst.working_tree_dir) / "README.txt").is_file()
  assert not (pathlib.Path(dst.working_tree_dir) / "src" / "main.c").is_file()
  assert not (pathlib.Path(dst.working_tree_dir) / "src" / "Makefile").is_file()


def test_ExportSubtree_integration_test_readme_and_makefile(
  repo_with_history: git.Repo, empty_repo: git.Repo
):
  src, dst = repo_with_history, empty_repo

  export_subtree.ExportSubtree(src, dst, {"README.txt", "src/Makefile"})

  dst_commits = export_subtree.GetCommitsInOrder(dst)
  assert (pathlib.Path(dst.working_tree_dir) / "README.txt").is_file()
  assert not (pathlib.Path(dst.working_tree_dir) / "src" / "main.c").is_file()
  assert not (pathlib.Path(dst.working_tree_dir) / "src" / "Makefile").is_file()
  assert len(dst_commits) == 3


def test_ExportSubtree_integration_test_nothing_interesting(
  repo_with_history: git.Repo, empty_repo: git.Repo
):
  src, dst = repo_with_history, empty_repo

  export_subtree.ExportSubtree(src, dst, {"foo"})

  dst_commits = export_subtree.GetCommitsInOrder(dst)
  assert len(dst_commits) == 0
  assert not (pathlib.Path(dst.working_tree_dir) / "README.txt").is_file()
  assert not (pathlib.Path(dst.working_tree_dir) / "src" / "main.c").is_file()
  assert not (pathlib.Path(dst.working_tree_dir) / "src" / "Makefile").is_file()


def test_ExportSubtree_duplicate_exports(
  repo_with_history: git.Repo, empty_repo: git.Repo
):
  src, dst = repo_with_history, empty_repo

  export_subtree.ExportSubtree(
    src, dst, {"README.txt", "src/main.c", "src/Makefile"}
  )
  commits = export_subtree.GetCommitsInOrder(dst)

  # Do it again.
  export_subtree.ExportSubtree(
    src, dst, {"README.txt", "src/main.c", "src/Makefile"}
  )
  commits2 = export_subtree.GetCommitsInOrder(dst)

  assert 2 * len(commits) == len(commits2)


def test_ExportSubtree_split_export(
  repo_with_history: git.Repo, empty_repo: git.Repo
):
  src, dst = repo_with_history, empty_repo

  export_subtree.ExportSubtree(
    src, dst, {"README.txt", "src/main.c", "src/Makefile"}, head_ref="HEAD~"
  )
  commits = export_subtree.GetCommitsInOrder(dst)

  # Do it again.
  export_subtree.ExportSubtree(
    src, dst, {"README.txt", "src/main.c", "src/Makefile"}
  )
  commits2 = export_subtree.GetCommitsInOrder(dst)

  assert (len(commits) * 2) + 1 == len(commits2)


@test.XFail(reason="fix me")
def test_ExportSubtree_dirty_destination(
  repo_with_history: git.Repo, empty_repo: git.Repo
):
  src, dst = repo_with_history, empty_repo
  fs.Write(
    pathlib.Path(dst.working_tree_dir) / "README.txt",
    "I'm dirty".encode("utf-8"),
  )
  with test.Raises(OSError) as e_ctx:
    export_subtree.ExportSubtree(src, dst, {"README.txt"})
  assert str(e_ctx.value).endswith(" is dirty")


if __name__ == "__main__":
  test.Main()
