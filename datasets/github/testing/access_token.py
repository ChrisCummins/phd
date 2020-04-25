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
"""This module exposes the path of an access token for testing, the access token
contents, if it exists.
"""
from datasets.github import api

ACCESS_TOKEN_PATH = api.TEST_ACCESS_TOKEN_PATH

if ACCESS_TOKEN_PATH.is_file():
  ACCESS_TOKEN = api.ReadGithubAccessTokenPath(ACCESS_TOKEN_PATH)
else:
  ACCESS_TOKEN = None
