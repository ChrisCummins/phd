# Fuzz testing Java using CLgen.

filegroup(
    name = "java",
    srcs = [
        "clgen.pbtxt",
        "clone_list.pbtxt",
    ],
)

py_test(
    name = "test_config_files_are_valid",
    srcs = ["test_config_files_are_valid.py"],
    data = [":java"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//deeplearning/clgen",
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//lib/labm8:bazelutil",
        "//lib/labm8:pbutil",
    ],
)
