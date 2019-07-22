load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_binary(
    name = "notebooks",
    srcs = ["//notebooks"],
    deps = [
        "//datasets/github/scrape_repos:contentfiles",
        "//deeplearning/clgen/corpuses:encoded",
        "//deeplearning/clgen/corpuses:preprocessed",
        "//deeplearning/clgen/proto:clgen_pb_py",
    ],
)

py3_image(
    name = "notebooks_image",
    srcs = [":notebooks"],
    base = "@base//image",
    main = "notebooks.py",
    deps = [
        ":notebooks",
    ],
)
