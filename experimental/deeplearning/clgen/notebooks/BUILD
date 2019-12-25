load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_binary(
    name = "notebooks",
    srcs = ["//notebooks"],
    deps = [
        "//deeplearning/clgen",
        "//experimental/deeplearning/clgen/backtracking:backtracking_db",
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//research/cummins_2017_cgo:generative_model",
    ],
)

py3_image(
    name = "image",
    srcs = [":notebooks"],
    base = "@phd_base//image",
    main = "notebooks.py",
    deps = [
        ":notebooks",
    ],
)
