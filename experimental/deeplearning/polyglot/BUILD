# Get a baseline reading from a CLgen model.

py_library(
    name = "get_instances",
    srcs = ["get_instances.py"],
    data = [":protos"],
    visibility = ["//experimental:__subpackages__"],
    deps = [
        "//deeplearning/clgen",
        "//lib/labm8:lockfile",
    ],
)

py_test(
    name = "get_instances_test",
    srcs = ["get_instances_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":get_instances",
    ],
)

py_binary(
    name = "notebooks",
    srcs = ["notebooks.py"],
    deps = [
        "//datasets/github/scrape_repos:contentfiles",
        "//deeplearning/clgen",
        "//experimental/deeplearning/polyglot:get_instances",
        "//lib/labm8:bazelutil",
        "//lib/labm8:pbutil",
        "//lib/labm8:viz",
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
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//experimental:__subpackages__"],
    deps = [
        ":get_instances",
        "//deeplearning/clgen",
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//lib/labm8:lockfile",
        "//lib/labm8:pbutil",
    ],
)

py_test(
    name = "test_protos_are_valid",
    srcs = ["test_protos_are_valid.py"],
    data = [":protos"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//lib/labm8:bazelutil",
        "//lib/labm8:pbutil",
    ],
)
