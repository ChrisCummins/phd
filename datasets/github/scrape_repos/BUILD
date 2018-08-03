# This package contains a tool for scraping repos from GitHub.

load("@requirements//:requirements.bzl", "requirement")

py_binary(
    name = "cloner",
    srcs = ["cloner.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//lib/labm8:fs",
        "//lib/labm8:pbutil",
        "//third_party/py/absl",
        requirement("humanize"),
        requirement("progressbar2"),
        requirement("python-utils"),
    ],
)

py_library(
    name = "contentfiles",
    srcs = ["contentfiles.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//lib/labm8:labdate",
        "//lib/labm8:sqlutil",
        requirement("sqlalchemy"),
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "export_corpus",
    srcs = ["export_corpus.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":contentfiles",
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//lib/labm8:pbutil",
        "//third_party/py/absl",
        requirement("humanize"),
        requirement("sqlalchemy"),
    ],
)

py_binary(
    name = "importer",
    srcs = ["importer.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":contentfiles",
        "//datasets/github/scrape_repos/preprocessors",
        "//datasets/github/scrape_repos/preprocessors:public",
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//lib/labm8:pbutil",
        "//third_party/py/absl",
        requirement("humanize"),
        requirement("progressbar2"),
        requirement("python-utils"),
        requirement("sqlalchemy"),
    ],
)

py_binary(
    name = "scraper",
    srcs = ["scraper.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//lib/labm8:fs",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
        "//third_party/py/absl",
        requirement("PyGithub"),
        requirement("humanize"),
        requirement("progressbar2"),
        requirement("python-utils"),
    ],
)

py_test(
    name = "scraper_test",
    srcs = ["scraper_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":scraper",
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//lib/labm8:labdate",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
