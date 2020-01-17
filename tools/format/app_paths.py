# Copyright 2020 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module defines accessors to filesystem paths."""
import os
import pathlib

import appdirs
import build_info
from labm8.py import app

FLAGS = app.FLAGS


def GetCacheDir() -> pathlib.Path:
  """Resolve the cache directory for linters."""
  _BAZEL_TEST_TMPDIR = os.environ.get("TEST_TMPDIR")
  if _BAZEL_TEST_TMPDIR:
    cache_dir = pathlib.Path(os.environ["TEST_TMPDIR"]) / "cache"
  else:
    cache_dir = pathlib.Path(
      appdirs.user_cache_dir(
        "phd_format", "Chris Cummins", version=build_info.GetBuildInfo().version
      )
    )
  cache_dir.mkdir(parents=True, exist_ok=True)
  return cache_dir
