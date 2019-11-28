# Get a baseline reading from a CLgen model.

py_binary(
    name = "get_instances",
    srcs = ["get_instances.py"],
    data = [":protos"],
    visibility = ["//experimental:__subpackages__"],
    deps = [
        "//deeplearning/clgen",
        "//labm8/py:app",
        "//labm8/py:lockfile",
    ],
)

py_test(
    name = "get_instances_test",
    size = "enormous",
    srcs = ["get_instances_test.py"],
    deps = [
        ":get_instances",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "notebooks",
    srcs = ["notebooks.py"],
    deps = [
        "//datasets/github/scrape_repos:contentfiles",
        "//deeplearning/clgen",
        "//experimental/deeplearning/polyglot:get_instances",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:pbutil",
        "//labm8/py:viz",
        "//third_party/py/jupyter",
    ],
)

filegroup(
    name = "protos",
    srcs = glob([
        "clone_lists/**/*.pbtxt",
        "corpuses/**/*.pbtxt",
        "models/**/*.pbtxt",
        "samplers/**/*.pbtxt",
    ]),
    visibility = ["//experimental:__subpackages__"],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    data = [":protos"],
    visibility = ["//experimental:__subpackages__"],
    deps = [
        ":get_instances",
        "//deeplearning/clgen",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:lockfile",
        "//labm8/py:pbutil",
    ],
)

py_test(
    name = "protos_are_valid_test",
    srcs = ["protos_are_valid_test.py"],
    data = [":protos"],
    deps = [
        "//datasets/github/scrape_repos/proto:scrape_repos_pb_py",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:pbutil",
        "//labm8/py:test",
    ],
)
