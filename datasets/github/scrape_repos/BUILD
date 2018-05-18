# This package contains a tool for scraping repos from GitHub.

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
    ],
)
