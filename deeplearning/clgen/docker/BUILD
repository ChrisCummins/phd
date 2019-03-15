# Docker images for CLgen.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py3_image(
    name = "clgen",
    srcs = ["clgen.py"],
    base = "@base//image",
    deps = [
        "//deeplearning/clgen",
        "//labm8:app",
    ],
)
