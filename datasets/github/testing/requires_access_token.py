# Copyright 2018-2020 Chris Cummins <chrisc.101@gmail.com>.
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
"""This module defines a decorator for marking skippable tests."""
from datasets.github.testing import access_token
from labm8.py import test

# Mark a test as skippable if there is no test access token for it to use. E.g.
#
#     from datasets.github.api.testing.requires_access_token import requires_access_token
#
#     @requires_access_token
#     def test_something_on_github():
#       github = api.GetDefaultGithubConnectionOrDie()
#       # go nuts ...
#
requires_access_token = test.SkipIf(
  not access_token.ACCESS_TOKEN_PATH.is_file(),
  reason="Test requires a Github access token",
)
