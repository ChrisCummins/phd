# This package contains CLgen, a deep learning program generator.
#
# CLgen is an open source application for generating runnable programs using
# deep learning. CLgen learns to program using neural networks which model the
# semantics and usage from large volumes of program fragments, generating
# executable programs that are representative of, but distinct from,
# the programs it learns from.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")
load("@requirements//:requirements.bzl", "requirement")

licenses(["restricted"])  # GPL v3

exports_files([
    "LICENSE",
    "README.md",
])

py_library(
    name = "cache",
    srcs = ["cache.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        "//lib/labm8:cache",
        "//lib/labm8:fs",
        requirement("absl-py"),
    ],
)

py_binary(
    name = "clgen",
    srcs = ["clgen.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":errors",
        ":samplers",
        "//deeplearning/clgen/models",
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//lib/labm8:pbutil",
        "//lib/labm8:prof",
        requirement("absl-py"),
    ],
)

py3_image(
    name = "clgen_image",
    srcs = ["clgen.py"],
    base = "@python3.6//image",
    main = "clgen.py",
    deps = [
        ":errors",
        ":samplers",
        "//deeplearning/clgen/models",
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//lib/labm8:pbutil",
        "//lib/labm8:prof",
        requirement("absl-py"),
    ],
)

py_test(
    name = "clgen_test",
    size = "enormous",
    srcs = ["clgen_test.py"],
    data = [
        "//deeplearning/clgen/tests/data:pico",
        "//deeplearning/clgen/tests/data:tiny",
        "//deeplearning/clgen/tests/data/tiny",
    ],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":clgen",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//lib/labm8:fs",
        "//lib/labm8:pbutil",
        "//lib/labm8:tar",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)

py_library(
    name = "conftest",
    testonly = True,
    srcs = ["conftest.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)

py_library(
    name = "errors",
    srcs = ["errors.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
)

py_binary(
    name = "ls_models",
    srcs = ["ls_models.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen/models",
        "//deeplearning/clgen/proto:internal_py_pb2",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
    ],
)

py_library(
    name = "samplers",
    srcs = ["samplers.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cache",
        ":errors",
        "//deeplearning/clgen/corpuses:atomizers",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//lib/labm8:crypto",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
    ],
)

py_test(
    name = "samplers_test",
    srcs = ["samplers_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":errors",
        ":samplers",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)

py_library(
    name = "telemetry",
    srcs = ["telemetry.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        "//deeplearning/clgen/proto:telemetry_py_pb2",
        "//lib/labm8:jsonutil",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
    ],
)

py_test(
    name = "telemetry_test",
    srcs = ["telemetry_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":telemetry",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)
