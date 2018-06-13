# Service processes for DeepSmith.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")
load("@requirements//:requirements.bzl", "requirement")

py_library(
    name = "services",
    srcs = ["services.py"],
    visibility = ["//deeplearning/deepsmith/cli:__subpackages__"],
    deps = [
        "//deeplearning/deepsmith/proto:service_py_pb2",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
        requirement("grpcio"),
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
        "//deeplearning/deepsmith/proto:datastore_py_pb2",
        "//lib/labm8:pbutil",
        requirement("grpcio"),
        requirement("absl-py"),
    ],
)

py3_image(
    name = "datastore_image",
    base = "@python3.6//image",
    srcs = ["datastore.py"],
    deps = [":datastore"],
    main = "datastore.py",
)

py_library(
    name = "generator",
    srcs = ["generator.py"],
    deps = [
        ":services",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//lib/labm8:pbutil",
    ],
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
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen/models",
        "//deeplearning/clgen/proto:model_py_pb2",
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

py_library(
    name = "harness",
    srcs = ["harness.py"],
    deps = [
        ":services",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//lib/labm8:pbutil",
    ],
)

py_binary(
    name = "cldrive",
    srcs = ["cldrive.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//deeplearning/deepsmith/cli:__subpackages__"],
    deps = [
        ":harness",
        ":services",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
        requirement("grpcio"),
    ],
)

py3_image(
    name = "cldrive_image",
    base = "@python3.6//image",
    srcs = ["cldrive.py"],
    deps = [":cldrive"],
    main = "cldrive.py",
)
