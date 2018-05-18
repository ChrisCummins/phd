py_library(
    name = "services",
    srcs = ["services.py"],
    visibility = ["//deeplearning/deepsmith/cli:__subpackages__"],
    deps = [
        "//deeplearning/deepsmith/proto:deepsmith",
        "//lib/labm8:pbutil",
    ],
)

py_library(
    name = "generator",
    srcs = ["generator.py"],
    deps = ["//deeplearning/deepsmith/proto:deepsmith", ":services"],
)

py_binary(
    name = "clgen",
    srcs = ["clgen.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//deeplearning/deepsmith/cli:__subpackages__"],
    deps = [
        ":generator",
        ":services",
        "//deeplearning/deepsmith/proto:deepsmith",
        "//lib/labm8:pbutil",
    ],
)

py_binary(
    name = "datastore",
    srcs = ["datastore.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//deeplearning/deepsmith/cli:__subpackages__"],
    deps = [
        ":services",
        "//deeplearning/deepsmith/proto:deepsmith",
        "//lib/labm8:pbutil",
    ],
)
