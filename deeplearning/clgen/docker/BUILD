# Docker images for CLgen.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py3_image(
    name = "clgen",
    srcs = ["//deeplearning/clgen"],
    base = "@base_java//image",
    visibility = ["//deeplearning/clgen/docker:__subpackages__"],
    deps = ["//deeplearning/clgen"],
)

py3_image(
    name = "clgen_preprocess",
    srcs = ["//deeplearning/clgen:preprocess"],
    base = "@base_java//image",
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
        "//labm8/py:dockerutil",
        "//labm8/py:fs",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "export_pretrained_model",
    srcs = ["export_pretrained_model.py"],
    deps = [
        "//deeplearning/clgen",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:pbutil",
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
        "//labm8/py:app",
        "//labm8/py:test",
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
