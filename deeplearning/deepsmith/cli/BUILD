# Copyright (c) 2017-2020 Chris Cummins.
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

py_binary(
    name = "explore",
    srcs = ["explore.py"],
    deps = [
        "//deeplearning/deepsmith:datastore",
        "//deeplearning/deepsmith:db",
        "//deeplearning/deepsmith:result",
        "//labm8/py:app",
        "//labm8/py:pbutil",
    ],
)

py_binary(
    name = "generate_testcases",
    srcs = ["generate_testcases.py"],
    deps = [
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:datastore_pb_py",
        "//deeplearning/deepsmith/proto:deepsmith_pb_py_grpc",
        "//deeplearning/deepsmith/proto:generator_pb_py_grpc",
        "//labm8/py:app",
    ],
)

py_binary(
    name = "import",
    srcs = ["import.py"],
    deps = [
        "//deeplearning/deepsmith:datastore",
        "//deeplearning/deepsmith:db",
        "//deeplearning/deepsmith:result",
        "//deeplearning/deepsmith:testcase",
        "//labm8/py:app",
        "//labm8/py:pbutil",
        "//third_party/py/progressbar",
    ],
)

py_binary(
    name = "run_testcases",
    srcs = ["run_testcases.py"],
    deps = [
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:datastore_pb_py",
        "//deeplearning/deepsmith/proto:deepsmith_pb_py_grpc",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//labm8/py:app",
    ],
)
