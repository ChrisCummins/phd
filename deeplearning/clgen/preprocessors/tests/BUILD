# Integration tests for CLgen preprocessing pipeline.

py_test(
    name = "benchmark_test",
    srcs = ["benchmark_test.py"],
    deps = [
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/preprocessors",
        "//deeplearning/clgen/preprocessors:common",
        "//deeplearning/clgen/preprocessors:cxx",
        "//deeplearning/clgen/preprocessors:opencl",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)
