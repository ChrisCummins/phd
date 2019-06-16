load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_binary(
    name = "export_random_contentfiles_subset",
    srcs = ["export_random_contentfiles_subset.py"],
    deps = [
        "//datasets/github/scrape_repos:contentfiles",
        "//labm8:app",
        "//labm8:humanize",
    ],
)

sh_test(
    name = "integration_test",
    srcs = ["integration_test.sh"],
    data = [
        ":export_random_contentfiles_subset",
        ":preprocess_java_corpus",
        ":scrape_java_files",
    ],
)

py_binary(
    name = "preprocess_java_corpus",
    srcs = ["preprocess_java_corpus.py"],
    deps = [
        "//datasets/github/scrape_repos:contentfiles",
        "//datasets/github/scrape_repos/preprocessors",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:humanize",
        "//labm8:sqlutil",
    ],
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
    deps = [":scrape_java_files"],
)
