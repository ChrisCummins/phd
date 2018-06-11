load("@requirements//:requirements.bzl", "requirement")

py_binary(
    name = "explore",
    srcs = ["explore.py"],
    deps = [
        "//deeplearning/deepsmith:datastore",
        "//deeplearning/deepsmith:db",
        "//deeplearning/deepsmith:result",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
    ],
)

py_binary(
    name = "generate_testcases",
    srcs = ["generate_testcases.py"],
    deps = [
        "//deeplearning/deepsmith/proto:datastore_py_pb2",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//deeplearning/deepsmith/services",
        requirement("absl-py"),
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
        "//lib/labm8:pbutil",
        requirement("absl-py"),
        requirement("progressbar2"),
    ],
)

py_binary(
    name = "run_testcases",
    srcs = ["run_testcases.py"],
    deps = [
        "//deeplearning/deepsmith/proto:datastore_py_pb2",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//deeplearning/deepsmith/services",
        requirement("absl-py"),
    ],
)
