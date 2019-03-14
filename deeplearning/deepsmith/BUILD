# DeepSmith is a framework for undirected fuzz testing.
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

licenses(["restricted"])  # GPL v3

exports_files(["LICENSE"])

py_library(
    name = "conftest",
    testonly = 1,
    srcs = ["conftest.py"],
    data = glob(["tests/data/datastores/*"]),
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        ":datastore",
        ":db",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//labm8:test",
    ],
)

py_library(
    name = "client",
    srcs = ["client.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [":db"],
)

py_library(
    name = "datastore",
    srcs = ["datastore.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":client",
        ":db",
        ":generator",
        ":harness",
        ":profiling_event",
        ":result",
        ":testbed",
        ":testcase",
        ":toolchain",
        "//deeplearning/deepsmith/proto:datastore_py_pb2",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//labm8:app",
        "//labm8:pbutil",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_library(
    name = "db",
    srcs = ["db.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/deepsmith/proto:datastore_py_pb2",
        "//labm8:app",
        "//labm8:labdate",
        "//labm8:pbutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "db_test",
    size = "small",
    srcs = ["db_test.py"],
    deps = [
        ":conftest",
        ":db",
        ":toolchain",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "generator",
    srcs = ["generator.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":db",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//labm8:labdate",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "generator_test",
    size = "small",
    srcs = ["generator_test.py"],
    deps = [
        ":conftest",
        ":generator",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "harness",
    srcs = ["harness.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":db",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//labm8:labdate",
        "//labm8:sqlutil",
        "//labm8:system",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "harness_test",
    size = "small",
    srcs = ["harness_test.py"],
    deps = [
        ":conftest",
        ":harness",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "toolchain",
    srcs = ["toolchain.py"],
    visibility = ["//visibility:public"],
    deps = [":db"],
)

py_library(
    name = "profiling_event",
    srcs = ["profiling_event.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":client",
        ":db",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//labm8:labdate",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_library(
    name = "result",
    srcs = ["result.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":db",
        ":profiling_event",
        ":testbed",
        ":testcase",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//labm8:labdate",
        "//labm8:pbutil",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "result_test",
    size = "small",
    srcs = ["result_test.py"],
    deps = [
        ":client",
        ":conftest",
        ":generator",
        ":harness",
        ":profiling_event",
        ":result",
        ":testbed",
        ":testcase",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//labm8:app",
        "//labm8:labdate",
        "//labm8:test",
    ],
)

py_library(
    name = "services",
    srcs = ["services.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/deepsmith/proto:service_py_pb2",
        "//labm8:app",
        "//labm8:pbutil",
        "//third_party/py/grpcio",
    ],
)

py_library(
    name = "testbed",
    srcs = ["testbed.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":db",
        ":toolchain",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//labm8:labdate",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "testbed_test",
    size = "small",
    srcs = ["testbed_test.py"],
    deps = [
        ":conftest",
        ":testbed",
        ":toolchain",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "testcase",
    srcs = ["testcase.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":db",
        ":generator",
        ":harness",
        ":profiling_event",
        ":toolchain",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//labm8:labdate",
        "//labm8:pbutil",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "testcase_test",
    size = "small",
    srcs = ["testcase_test.py"],
    deps = [
        ":client",
        ":conftest",
        ":generator",
        ":harness",
        ":profiling_event",
        ":testcase",
        ":toolchain",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//labm8:app",
        "//labm8:labdate",
        "//labm8:test",
    ],
)
