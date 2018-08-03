# This package contains a tool for scraping repos from GitHub.

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
        "//third_party/py/humanize",
        "//third_party/py/progressbar",
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
        "//third_party/py/absl",
        "//third_party/py/sqlalchemy",
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
        "//third_party/py/humanize",
        "//third_party/py/sqlalchemy",
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
        "//third_party/py/humanize",
        "//third_party/py/progressbar",
        "//third_party/py/sqlalchemy",
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
        "//third_party/py/github",
        "//third_party/py/humanize",
        "//third_party/py/progressbar",
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
