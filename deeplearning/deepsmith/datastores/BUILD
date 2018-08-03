# Datastore implementations.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")
load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//experimental:__subpackages__"])

py_binary(
    name = "datastore",
    srcs = ["datastore.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//deeplearning/deepsmith/cli:__subpackages__",
        "//experimental:__subpackages__",
    ],
    deps = [
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:datastore_py_pb2",
        "//lib/labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/grpcio",
    ],
)

py3_image(
    name = "datastore_image",
    srcs = ["datastore.py"],
    base = "@python3.6//image",
    main = "datastore.py",
    deps = [":datastore"],
)
