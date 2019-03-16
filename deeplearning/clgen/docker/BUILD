# Docker images for CLgen.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py3_image(
    name = "clgen",
    srcs = ["clgen.py"],
    base = "@base//image",
    visibility = ["//deeplearning/clgen/docker:__subpackages__"],
    deps = [
        "//deeplearning/clgen",
        "//labm8:app",
    ],
)

py_binary(
    name = "export_pretrained_model",
    srcs = ["export_pretrained_model.py"],
    deps = [
        "//deeplearning/clgen",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:pbutil",
    ],
)
