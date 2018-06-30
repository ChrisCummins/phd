load("@requirements//:requirements.bzl", "requirement")

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
        requirement("pytest"),
        requirement("pytest-benchmark"),
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
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
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
        "//lib/labm8:pbutil",
        "//lib/labm8:sqlutil",
        requirement("absl-py"),
        requirement("sqlalchemy"),
    ],
)

py_library(
    name = "db",
    srcs = ["db.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        "//deeplearning/deepsmith/proto:datastore_py_pb2",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
        requirement("sqlalchemy"),
    ],
)

py_test(
    name = "db_test",
    size = "small",
    srcs = ["db_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":db",
        ":toolchain",
        requirement("absl-py"),
        requirement("pytest"),
        requirement("pytest-benchmark"),
    ],
)

py_library(
    name = "generator",
    srcs = ["generator.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        ":db",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//lib/labm8:labdate",
        "//lib/labm8:sqlutil",
        requirement("sqlalchemy"),
    ],
)

py_test(
    name = "generator_test",
    size = "small",
    srcs = ["generator_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":generator",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        requirement("absl-py"),
        requirement("pytest"),
        requirement("pytest-benchmark"),
    ],
)

py_library(
    name = "harness",
    srcs = ["harness.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        ":db",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//lib/labm8:labdate",
        "//lib/labm8:sqlutil",
        "//lib/labm8:system",
        requirement("sqlalchemy"),
    ],
)

py_test(
    name = "harness_test",
    size = "small",
    srcs = ["harness_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":harness",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        requirement("absl-py"),
        requirement("pytest"),
        requirement("pytest-benchmark"),
    ],
)

py_library(
    name = "toolchain",
    srcs = ["toolchain.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [":db"],
)

py_library(
    name = "profiling_event",
    srcs = ["profiling_event.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        ":client",
        ":db",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//lib/labm8:labdate",
        "//lib/labm8:sqlutil",
        requirement("sqlalchemy"),
    ],
)

py_library(
    name = "result",
    srcs = ["result.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        ":db",
        ":profiling_event",
        ":testbed",
        ":testcase",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
        "//lib/labm8:sqlutil",
        requirement("sqlalchemy"),
    ],
)

py_test(
    name = "result_test",
    size = "small",
    srcs = ["result_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
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
        "//lib/labm8:labdate",
        requirement("absl-py"),
        requirement("pytest"),
        requirement("pytest-benchmark"),
    ],
)

py_library(
    name = "services",
    srcs = ["services.py"],
    visibility = [
        "//deeplearning/deepsmith:__subpackages__",
        "//experimental:__subpackages__",
    ],
    deps = [
        "//deeplearning/deepsmith/proto:service_py_pb2",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
        requirement("grpcio"),
    ],
)

py_library(
    name = "testbed",
    srcs = ["testbed.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        ":db",
        ":toolchain",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//lib/labm8:labdate",
        "//lib/labm8:sqlutil",
        requirement("sqlalchemy"),
    ],
)

py_test(
    name = "testbed_test",
    size = "small",
    srcs = ["testbed_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":testbed",
        ":toolchain",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        requirement("absl-py"),
        requirement("pytest"),
        requirement("pytest-benchmark"),
    ],
)

py_library(
    name = "testcase",
    srcs = ["testcase.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        ":db",
        ":generator",
        ":harness",
        ":profiling_event",
        ":toolchain",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
        "//lib/labm8:sqlutil",
        requirement("sqlalchemy"),
    ],
)

py_test(
    name = "testcase_test",
    size = "small",
    srcs = ["testcase_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":client",
        ":conftest",
        ":generator",
        ":harness",
        ":profiling_event",
        ":testcase",
        ":toolchain",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//lib/labm8:labdate",
        requirement("absl-py"),
        requirement("pytest"),
        requirement("pytest-benchmark"),
    ],
)
