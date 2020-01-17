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
"""This module defines a formatter for java sources."""
from labm8.py import bazelutil
from tools.format.formatters.base import batched_file_formatter


class FormatJava(batched_file_formatter.BatchedFileFormatter):
  """Format Java sources."""

  def __init__(self, *args, **kwargs):
    super(FormatJava, self).__init__(*args, **kwargs)
    self.java = self._Which("java")

    self.google_java_format = bazelutil.DataPath(
      "phd/third_party/java/google-java-format-1.7-all-deps.jar"
    )

  def RunMany(self, paths):
    return self._Exec(
      [self.java, "-jar", self.google_java_format, "-i"] + paths
    )
