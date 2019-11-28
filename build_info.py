# Copyright 2019 Chris Cummins <chrisc.101@gmail.com>.
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
"""Access to the build information."""
import datetime
import functools
import re
import typing

import build_info_pbtxt_py
import config_pb2
import version_py

from labm8.py import pbutil


@functools.lru_cache()
def GetBuildInfo() -> config_pb2.BuildInfo:
  """Return the build state."""
  return pbutil.FromString(
    build_info_pbtxt_py.STRING, config_pb2.BuildInfo(), uninitialized_okay=False
  )


def GetGithubCommitUrl(
  remote_url: typing.Optional[str] = None,
  commit_hash: typing.Optional[str] = None,
) -> typing.Optional[str]:
  """Calculate the GitHub URL for a commit."""
  try:
    build_info = GetBuildInfo()
  except OSError:
    return "https://github.com/ChrisCummins/phd"
  remote_url = remote_url or build_info.git_remote_url
  commit_hash = commit_hash or build_info.git_commit

  m = re.match(f"git@github\.com:([^/]+)/(.+)\.git", remote_url)
  if not m:
    return None
  return f"https://github.com/{m.group(1)}/{m.group(2)}/commit/{commit_hash}"


def FormatShortRevision(html: bool = False) -> str:
  """Get a shortened revision string."""
  build_info = GetBuildInfo()
  dirty_suffix = "*" if build_info.repo_dirty else ""
  short_hash = f"{build_info.git_commit[:7]}{dirty_suffix}"
  if html:
    return f'<a href="{GetGithubCommitUrl()}">{short_hash}</a>'
  else:
    return short_hash


def FormatVersion() -> str:
  return f"version: {Version()}"


def Version() -> str:
  return version_py.STRING.strip()


def BuildTimestamp() -> int:
  build_info = GetBuildInfo()
  return build_info.seconds_since_epoch


def FormatShortBuildDescription(html: bool = False) -> str:
  """Get build string in the form: 'build SHORT_HASH on DATE by USER@HOST'."""
  build_info = GetBuildInfo()
  natural_date = datetime.datetime.fromtimestamp(
    build_info.seconds_since_epoch
  ).strftime("%Y-%m-%d")
  revision = FormatShortRevision(html)
  return (
    f"build: {revision} on {natural_date} by "
    f"{build_info.user}@{build_info.host}"
  )


def FormatLongBuildDescription(html: bool = False) -> str:
  """Get long multi-line build string."""
  build_info = GetBuildInfo()
  natural_datetime = datetime.datetime.fromtimestamp(
    build_info.seconds_since_epoch
  ).strftime("%Y-%m-%d %H:%M:%S")
  revision = FormatShortRevision(html=html)
  return f"""\
Built by {build_info.user}@{build_info.host} at {natural_datetime}.
Revision: {revision}."""
