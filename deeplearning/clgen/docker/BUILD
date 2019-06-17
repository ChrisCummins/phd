# Docker images for CLgen.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py3_image(
    name = "clgen",
    srcs = ["//deeplearning/clgen"],
    base = "@base//image",
    visibility = ["//deeplearning/clgen/docker:__subpackages__"],
    deps = ["//deeplearning/clgen"],
)

py3_image(
    name = "clgen_preprocess",
    srcs = ["//deeplearning/clgen:preprocess"],
    base = "@base//image",
    main = "preprocess.py",
    visibility = ["//deeplearning/clgen/docker:__subpackages__"],
    deps = ["//deeplearning/clgen:preprocess"],
)

py_test(
    name = "clgen_preprocess_test",
    srcs = ["clgen_preprocess_test.py"],
    data = [
        ":clgen_preprocess.tar",
    ],
    deps = [
        "//labm8:dockerutil",
        "//labm8:fs",
        "//labm8:test",
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
        "//research/cummins_2017_cgo:generative_model",
    ],
)

py_test(
    name = "export_pretrained_model_test",
    srcs = ["export_pretrained_model_test.py"],
    deps = [
        ":export_pretrained_model",
        "//deeplearning/clgen",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)

py3_image(
    name = "export_pretrained_model_image",
    srcs = [":export_pretrained_model"],
    base = "@base//image",
    main = "export_pretrained_model.py",
    deps = [
        ":export_pretrained_model",
    ],
)
