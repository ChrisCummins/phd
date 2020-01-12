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
from tools.format import formatter


class FormatJava(formatter.BatchedFormatter):
  """Format Java sources."""

  def __init__(self, *args, **kwargs):
    super(FormatJava, self).__init__(*args, **kwargs)
    self.java = formatter.WhichOrDie("java")

    # Unpack the jarfile to the local cache. We do this rather than accessing
    # the data file directly since a par build embeds the data inside the
    # package. See: github.com/google/subpar/issues/43
    self.google_java_format = (
      self.cache_path / "google-java-format-1.7-all-deps.jar"
    )
    if not self.google_java_format.is_file():
      jar = bazelutil.DataString(
        "phd/third_party/java/google-java-format-1.7-all-deps.jar"
      )
      with open(self.google_java_format, "wb") as f:
        f.write(jar)

  def RunMany(self, paths):
    return formatter.ExecOrError(
      [self.java, "-jar", self.google_java_format, "-i"] + paths
    )
