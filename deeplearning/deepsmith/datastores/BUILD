# Datastore implementations.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

package(default_visibility = ["//experimental:__subpackages__"])

py_binary(
    name = "datastore",
    srcs = ["datastore.py"],
    visibility = [
        "//deeplearning/deepsmith/cli:__subpackages__",
        "//experimental:__subpackages__",
    ],
    deps = [
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:datastore_py_pb2",
        "//labm8:app",
        "//labm8:pbutil",
        "//third_party/py/grpcio",
    ],
)

py3_image(
    name = "datastore_image",
    srcs = ["datastore.py"],
    base = "@base//image",
    main = "datastore.py",
    deps = [":datastore"],
)
