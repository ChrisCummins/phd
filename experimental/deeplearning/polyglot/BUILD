# Get a baseline reading from a CLgen model.

py_binary(
    name = "get_instances",
    srcs = ["get_instances.py"],
    data = [":protos"],
    visibility = ["//experimental:__subpackages__"],
    deps = [
        "//deeplearning/clgen",
        "//labm8:lockfile",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "get_instances_test",
    size = "enormous",
    srcs = ["get_instances_test.py"],
    deps = [
        ":get_instances",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "notebooks",
    srcs = ["notebooks.py"],
    deps = [
        "//datasets/github/scrape_repos:contentfiles",
        "//deeplearning/clgen",
        "//experimental/deeplearning/polyglot:get_instances",
        "//labm8:bazelutil",
        "//labm8:pbutil",
        "//labm8:viz",
        "//third_party/py/absl",
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
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//labm8:humanize",
        "//labm8:lockfile",
        "//labm8:pbutil",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "protos_are_valid_test",
    srcs = ["protos_are_valid_test.py"],
    data = [":protos"],
    deps = [
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//labm8:bazelutil",
        "//labm8:pbutil",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)
