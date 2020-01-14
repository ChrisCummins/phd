load(
    "//third_party/pybind11_bazel:build_defs.bzl",
    "pybind_extension",
)

pybind_extension(
    "example",
    srcs = ["example.cc"],
    copts = ["-DPYTHON_EXECUTABLE=/usr/bin/python3"],
)

py_test(
    name = "example_test",
    srcs = ["example_test.py"],
    data = [":example.so"],
    deps = [
        "//labm8/py:test",
    ],
)
