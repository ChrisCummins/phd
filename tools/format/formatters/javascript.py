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
"""This module defines a formatter for JavaScript, HTML, and CSS sources."""
from tools.format import formatter


class FormatJavaScript(formatter.BatchedFormatter):
  """Format Javascript / HTML / CSS sources."""

  def __init__(self, *args, **kwargs):
    super(FormatJavaScript, self).__init__(*args, **kwargs)
    self.js_beautify = formatter.WhichOrDie("js-beautify")

  def RunMany(self, paths):
    return formatter.ExecOrError(
      [self.js_beautify, "--replace", "--config", JSBEAUTIFY_RC] + paths
    )
