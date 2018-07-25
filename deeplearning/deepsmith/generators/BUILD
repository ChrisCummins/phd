# Generator implementations.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")
load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//experimental:__subpackages__"])

py_binary(
    name = "clgen",
    srcs = ["clgen.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":clgen_pretrained",
        "//deeplearning/clgen",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        requirement("grpcio"),
        requirement("absl-py"),
    ],
)

py_binary(
    name = "clgen_pretrained",
    srcs = ["clgen_pretrained.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":generator",
        "//deeplearning/clgen:sample",
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        requirement("grpcio"),
        requirement("absl-py"),
    ],
)

# TODO(cec): Re-enable once CLgen cyclic symlinks is fixed.
# py3_image(
#     name = "clgen_image",
#     base = "@python3.6//image",
#     srcs = ["clgen.py"],
#     deps = [":clgen"],
#     main = "clgen.py",
# )

py_binary(
    name = "clsmith",
    srcs = ["clsmith.py"],
    data = ["@CLSmith"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":generator",
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//deeplearning/deepsmith/proto:service_py_pb2",
        "//lib/labm8:bazelutil",
        "//lib/labm8:labdate",
        requirement("grpcio"),
        requirement("absl-py"),
    ],
)

py_test(
    name = "clsmith_test",
    srcs = ["clsmith_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":clsmith",
        requirement("pytest"),
        requirement("absl-py"),
    ],
)

py_binary(
    name = "dummy_generator",
    srcs = ["dummy_generator.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//deeplearning/deepsmith/cli:__subpackages__",
        "//experimental:__subpackages__",
    ],
    deps = [
        ":generator",
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
        requirement("grpcio"),
        requirement("absl-py"),
    ],
)

py_library(
    name = "generator",
    srcs = ["generator.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//lib/labm8:pbutil",
    ],
)

py_binary(
    name = "randchar",
    srcs = ["randchar.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":generator",
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
        requirement("grpcio"),
        requirement("absl-py"),
    ],
)
