py_library(
    name = "services",
    srcs = ["services.py"],
    deps = ["//lib/labm8:pbutil"],
)

py_binary(
    name = "clgen",
    srcs = ["clgen.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":services",
        "//deeplearning/deepsmith/proto:deepsmith",
        "//lib/labm8:pbutil",
    ],
)
