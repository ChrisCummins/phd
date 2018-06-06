# Jupyter notebooks server.

py_binary(
    name = "notebooks",
    srcs = ["notebooks.py"],
    data = ["//experimental/polyglot/baselines:protos"],
    deps = [
        "//deeplearning/clgen",
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//experimental/polyglot/baselines:get_instances",
        "//lib/labm8:bazelutil",
        "//lib/labm8:pbutil",
        "//lib/labm8:viz",
    ],
)
