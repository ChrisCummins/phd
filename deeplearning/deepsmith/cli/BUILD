py_binary(
    name = "import",
    srcs = ["import.py"],
    deps = [
        "//deeplearning/deepsmith:datastore",
        "//deeplearning/deepsmith:db",
        "//deeplearning/deepsmith:result",
        "//deeplearning/deepsmith:testcase",
        "//deeplearning/deepsmith/proto:deepsmith",
        "//lib/labm8:pbutil",
    ],
)

py_binary(
    name = "explore",
    srcs = ["explore.py"],
    deps = [
        "//deeplearning/deepsmith:datastore",
        "//deeplearning/deepsmith:db",
        "//deeplearning/deepsmith:result",
        "//deeplearning/deepsmith/proto:deepsmith",
        "//lib/labm8:pbutil",
    ],
)

py_binary(
    name = "generate_testcases",
    srcs = ["generate_testcases.py"],
    deps = [
        "//deeplearning/deepsmith/proto:deepsmith",
        "//deeplearning/deepsmith/services",
    ],
)

py_binary(
    name = "run_testcases",
    srcs = ["run_testcases.py"],
    deps = [
        "//deeplearning/deepsmith/proto:deepsmith",
        "//deeplearning/deepsmith/services",
    ],
)
