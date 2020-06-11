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
"""Unit tests for //tools/format/formatters:bazel."""
from labm8.py import test
from tools.format.formatters import bazel

FLAGS = test.FLAGS


def test_small_build_file():
  text = bazel.FormatBuild.Format(
    """
py_binary(
name = "foo",
srcs = ["foo.py"],
deps = [":b", ":a", "//third_party/foo",]
)
"""
  )
  print(text)
  assert (
    text
    == """\
py_binary(
    name = "foo",
    srcs = ["foo.py"],
    deps = [":b", ":a", "//third_party/foo"],
)
"""
  )


def test_build_file_without_trailing_newline():
  text = bazel.FormatBuild.Format(
    """cc_binary(
  name = "foo"    , srcs = ["foo.cc"])"""
  )
  print(text)
  assert (
    text
    == """\
cc_binary(
    name = "foo",
    srcs = ["foo.cc"],
)
"""
  )


if __name__ == "__main__":
  test.Main()
