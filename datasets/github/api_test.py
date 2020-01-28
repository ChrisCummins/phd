# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for //datasets/github:api."""
import os
import pathlib

from datasets.github import api
from datasets.github.testing.requires_access_token import requires_access_token
from labm8.py import app
from labm8.py import fs
from labm8.py import test

FLAGS = app.FLAGS


def test_ReadGithubAccessTokenPath_file_not_found(tempdir: pathlib.Path):
  with test.Raises(FileNotFoundError):
    api.ReadGithubAccessTokenPath(tempdir / "not_a_file.txt")


def test_ReadGithubAccessTokenPath_invalid_empty_file(tempdir: pathlib.Path):
  path = tempdir / "access_token.txt"
  path.touch()
  with test.Raises(api.BadCredentials) as e_ctx:
    api.ReadGithubAccessTokenPath(path)

  assert str(e_ctx.value) == "Access token not found in file"


def test_ReadGithubAccessTokenPath_invalid_no_permissions(
  tempdir: pathlib.Path,
):
  path = tempdir / "access_token.txt"
  path.touch()
  os.chmod(path, 0o000)
  with test.Raises(api.BadCredentials) as e_ctx:
    api.ReadGithubAccessTokenPath(path)

  assert str(e_ctx.value) == "Cannot read file"


def test_ReadGithubAccessTokenPath_invalid_is_directory(tempdir: pathlib.Path,):
  path = tempdir / "access_token.txt"
  path.mkdir()
  with test.Raises(api.BadCredentials) as e_ctx:
    api.ReadGithubAccessTokenPath(path)

  assert str(e_ctx.value) == "File is a directory"


def test_ReadGithubAccessTokenPath_valid_file(tempdir: pathlib.Path,):
  path = tempdir / "access_token.txt"
  fs.Write(path, "1234".encode("utf-8"))

  assert api.ReadGithubAccessTokenPath(path) == "1234"


def test_GetDefaultGithubAccessToken_from_GITHUB_ACCESS_TOKEN(
  tempdir: pathlib.Path,
):
  with test.TemporaryEnv() as env:
    env["GITHUB_ACCESS_TOKEN"] = "1234"
    source, token = api.GetDefaultGithubAccessToken()

  assert source == "$GITHUB_ACCESS_TOKEN"
  assert token == "1234"


def test_GetDefaultGithubAccessToken_from_GITHUB_ACCESS_TOKEN_PATH(
  tempdir: pathlib.Path,
):
  path = tempdir / "access_token_path.txt"
  fs.Write(path, "1234".encode("utf-8"))
  with test.TemporaryEnv() as env:
    env["GITHUB_ACCESS_TOKEN_PATH"] = str(path)
    source, token = api.GetDefaultGithubAccessToken()

  assert source == f"$GITHUB_ACCESS_TOKEN_PATH={path}"
  assert token == "1234"


def test_GetDefaultGithubAccessToken_from_GITHUB_ACCESS_TOKEN_PATH_invalid_file(
  tempdir: pathlib.Path,
):
  path = tempdir / "access_token_path.txt"
  path.touch()
  with test.TemporaryEnv() as env:
    env["GITHUB_ACCESS_TOKEN_PATH"] = str(path)
    with test.Raises(api.BadCredentials) as e_ctx:
      api.GetDefaultGithubAccessToken()

  assert str(e_ctx.value) == (
    f"Invalid credentials file $GITHUB_ACCESS_TOKEN_PATH={path}: "
    "Access token not found in file"
  )


def test_GetDefaultGithubAccessToken_from_github_access_token_flag(
  tempdir: pathlib.Path,
):
  FLAGS.unparse_flags()
  FLAGS(["argv[0]", "--github_access_token", "1234"])

  source, token = api.GetDefaultGithubAccessToken()

  assert source == "--github_access_token"
  assert token == "1234"


def test_GetDefaultGithubAccessToken_from_github_access_token_path_flag(
  tempdir: pathlib.Path,
):
  path = tempdir / "access_token_path.txt"
  fs.Write(path, "1234".encode("utf-8"))

  FLAGS.unparse_flags()
  FLAGS(["argv[0]", "--github_access_token_path", str(path)])

  source, token = api.GetDefaultGithubAccessToken()

  assert source == f"--github_access_token_path={path}"
  assert token == "1234"


def test_GetDefaultGithubAccessToken_invalid_github_access_token_path_flag(
  tempdir: pathlib.Path,
):
  path = tempdir / "access_token_path.txt"
  path.touch()

  FLAGS.unparse_flags()
  FLAGS(["argv[0]", "--github_access_token_path", str(path)])

  with test.Raises(api.BadCredentials) as e_ctx:
    api.GetDefaultGithubAccessToken()

  assert str(e_ctx.value) == (
    f"Invalid credentials file --github_access_token_path={path}: "
    "Access token not found in file"
  )


def test_GetDefaultGithubAccessToken_from_extra_paths(tempdir: pathlib.Path):
  path = tempdir / "access_token_path.txt"
  fs.Write(path, "1234".encode("utf-8"))

  source, token = api.GetDefaultGithubAccessToken(
    extra_access_token_paths=[path]
  )

  assert source == str(path)
  assert token == "1234"


@requires_access_token
def test_GetDefaultGithubConnection():
  github = api.GetDefaultGithubConnection()
  github.get_user("ChrisCummins")


@requires_access_token
def test_GetDefaultGithubConnectionOrDie():
  github = api.GetDefaultGithubConnectionOrDie()
  github.get_user("ChrisCummins")


@requires_access_token
@test.Parametrize(
  "shallow", (False, True), names=("deep_clone", "shallow_clone")
)
def test_CloneRepo_valid_repo(tempdir: pathlib.Path, shallow: bool):
  github = api.GetDefaultGithubConnectionOrDie()
  repo = github.get_repo("ChrisCummins/empty_repository_for_testing")
  clone_path = tempdir / "repo"

  # Note forced https because test runner may not have access to SSH
  # keys in ~/.ssh.
  assert (
    api.CloneRepo(repo, tempdir / "repo", shallow=shallow, force_https=True)
    == clone_path
  )
  assert (clone_path / "HelloWorld.java").is_file()


@requires_access_token
@test.Parametrize(
  "shallow", (False, True), names=("deep_clone", "shallow_clone")
)
def test_CloneRepo_invalid_repo_not_found(tempdir: pathlib.Path, shallow: bool):
  github = api.GetDefaultGithubConnectionOrDie()
  repo = github.get_repo("ChrisCummins/not_a_real_repo")

  with test.Raises(FileNotFoundError):
    api.CloneRepo(repo, tempdir, shallow=shallow)


if __name__ == "__main__":
  test.Main()
