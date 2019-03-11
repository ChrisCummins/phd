py_binary(
    name = "explore",
    srcs = ["explore.py"],
    deps = [
        "//deeplearning/deepsmith:datastore",
        "//deeplearning/deepsmith:db",
        "//deeplearning/deepsmith:result",
        "//labm8:app",
        "//labm8:pbutil",
    ],
)

py_binary(
    name = "generate_testcases",
    srcs = ["generate_testcases.py"],
    deps = [
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:datastore_py_pb2",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//labm8:app",
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
        "//labm8:app",
        "//labm8:pbutil",
        "//third_party/py/progressbar",
    ],
)

py_binary(
    name = "run_testcases",
    srcs = ["run_testcases.py"],
    deps = [
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:datastore_py_pb2",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//labm8:app",
    ],
)
