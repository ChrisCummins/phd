# Differential tests for DeepSmith.
#
# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepSmith is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSmith is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepSmith.  If not, see <https://www.gnu.org/licenses/>.

py_library(
    name = "difftests",
    srcs = ["difftests.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/deepsmith/proto:deepsmith_pb_py_grpc",
        "//labm8:app",
    ],
)

py_test(
    name = "difftests_test",
    srcs = ["difftests_test.py"],
    deps = [
        ":difftests",
        "//deeplearning/deepsmith/proto:deepsmith_pb_py_grpc",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "opencl",
    srcs = ["opencl.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":difftests",
        "//deeplearning/deepsmith/proto:deepsmith_pb_py_grpc",
        "//gpu/cldrive/legacy:args",
        "//labm8:app",
    ],
)
