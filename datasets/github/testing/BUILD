# Helpers for for writing tests that interface with Github API.
#
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

py_library(
    name = "access_token",
    testonly = 1,
    srcs = ["access_token.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//datasets/github:api",
    ],
)

py_library(
    name = "requires_access_token",
    testonly = 1,
    srcs = ["requires_access_token.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":access_token",
        "//labm8/py:test",
    ],
)
