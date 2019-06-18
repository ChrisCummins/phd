load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_binary(
    name = "export_java_corpus",
    srcs = ["export_java_corpus.py"],
    deps = [
        "//datasets/github/scrape_repos:contentfiles",
        "//datasets/github/scrape_repos/preprocessors",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:humanize",
        "//labm8:sqlutil",
    ],
)

py3_image(
    name = "export_java_corpus_image",
    srcs = ["export_java_corpus.py"],
    base = "@base//image",
    main = "export_java_corpus.py",
    visibility = ["//experimental/deeplearning/deepsmith/java_fuzz:__subpackages__"],
    deps = [":export_java_corpus"],
)

py_test(
    name = "export_java_corpus_test",
    srcs = ["export_java_corpus_test.py"],
    deps = [
        ":export_java_corpus",
        "//datasets/github/scrape_repos:contentfiles",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "mask_contentfiles",
    srcs = ["mask_contentfiles.py"],
    deps = [
        "//datasets/github/scrape_repos:contentfiles",
        "//labm8:app",
        "//labm8:humanize",
        "//third_party/py/sqlalchemy",
    ],
)

py3_image(
    name = "mask_contentfiles_image",
    srcs = ["mask_contentfiles.py"],
    base = "@base//image",
    main = "mask_contentfiles.py",
    visibility = ["//experimental/deeplearning/deepsmith/java_fuzz:__subpackages__"],
    deps = [":mask_contentfiles"],
)

py_binary(
    name = "scrape_java_files",
    srcs = ["scrape_java_files.py"],
    deps = [
        "//datasets/github:api",
        "//datasets/github/scrape_repos:contentfiles",
        "//datasets/github/scrape_repos:pipelined_scraper",
        "//datasets/github/scrape_repos/proto:scrape_repos_pb_py",
        "//labm8:app",
        "//third_party/py/github",
        "//third_party/py/progressbar",
        "//third_party/py/urllib3",
    ],
)

py3_image(
    name = "scrape_java_files_image",
    srcs = ["scrape_java_files.py"],
    base = "@base//image",
    main = "scrape_java_files.py",
    visibility = ["//experimental/deeplearning/deepsmith/java_fuzz:__subpackages__"],
    deps = [":scrape_java_files"],
)
