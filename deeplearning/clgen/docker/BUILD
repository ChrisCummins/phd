# Docker images for CLgen.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py3_image(
    name = "clgen",
    srcs = ["//deeplearning/clgen"],
    base = "@base//image",
    visibility = ["//deeplearning/clgen/docker:__subpackages__"],
    deps = ["//deeplearning/clgen"],
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

py3_image(
    name = "export_pretrained_model_image",
    srcs = ["export_pretrained_model.py"],
    base = "@base//image",
    main = "export_pretrained_model.py",
    deps = [
        "//deeplearning/clgen",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:pbutil",
    ],
)
