# Library for accessing the global configuration.
#
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

licenses(["notice"])  # Apache 2.0

py_library(
    name = "build_info",
    srcs = ["build_info.py"],
    data = [":build_info_proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//config/proto:config",
        "//labm8:bazelutil",
        "//labm8:pbutil",
    ],
)

py_test(
    name = "build_info_test",
    srcs = ["build_info_test.py"],
    deps = [
        ":build_info",
        "//labm8:test",
    ],
)

py_library(
    name = "getconfig",
    srcs = ["getconfig.py"],
    data = ["//:config"],
    visibility = ["//visibility:public"],
    deps = [
        "//config/proto:config",
        "//labm8:pbutil",
    ],
)

py_test(
    name = "getconfig_test",
    srcs = ["getconfig_test.py"],
    deps = [
        ":getconfig",
        "//labm8:app",
        "//labm8:test",
    ],
)

genrule(
    name = "build_info_proto",
    outs = ["build_info.pbtxt"],
    cmd = "$(location :make_build_info_proto) > $@",
    stamp = 1,
    tools = [":make_build_info_proto"],
)

sh_binary(
    name = "make_build_info_proto",
    srcs = ["make_build_info_proto.sh"],
)
