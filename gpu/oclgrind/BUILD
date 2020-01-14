# A Python interface to Oclgrind.
# See: https://github.com/jrprice/Oclgrind
#
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

load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_test")

py_binary(
    name = "oclgrind",
    srcs = ["oclgrind.py"],
    data = [
        "//third_party/oclgrind",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//gpu/clinfo/proto:clinfo_pb_py",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:system",
    ],
)

py_test(
    name = "oclgrind_test",
    srcs = ["oclgrind_test.py"],
    data = ["//gpu/oclgrind/test/data:oclgrind_working"],
    deps = [
        ":oclgrind",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:test",
    ],
)

bats_test(
    name = "oclgrind_smoke_test",
    srcs = ["oclgrind_smoke_test.bats"],
    data = [
        "//gpu/clinfo",
        "//gpu/oclgrind/test/data:oclgrind_working",
        "//labm8/sh:test",
        "//third_party/oclgrind",
    ],
)
