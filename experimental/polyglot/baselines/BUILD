# Get a baseline reading from a CLgen model.

filegroup(
    name = "protos",
    srcs = glob([
        "corpuses/**/*.pbtxt",
        "models/**/*.pbtxt",
        "samplers/**/*.pbtxt",
    ]),
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    data = [":protos"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//deeplearning/clgen",
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//lib/labm8:pbutil",
    ],
)
