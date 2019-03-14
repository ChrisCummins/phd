# This package contains a tool for scraping repos from GitHub.
#
# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

py_binary(
    name = "cloner",
    srcs = ["cloner.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:humanize",
        "//labm8:pbutil",
        "//third_party/py/progressbar",
    ],
)

py_library(
    name = "contentfiles",
    srcs = ["contentfiles.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//labm8:app",
        "//labm8:labdate",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "export_corpus",
    srcs = ["export_corpus.py"],
    deps = [
        ":contentfiles",
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:pbutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_library(
    name = "github_repo",
    srcs = ["github_repo.py"],
    deps = [
        "//datasets/github/scrape_repos/preprocessors",
        "//datasets/github/scrape_repos/preprocessors:public",
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//labm8:app",
        "//labm8:pbutil",
        "//third_party/py/progressbar",
    ],
)

py_test(
    name = "github_repo_test",
    srcs = ["github_repo_test.py"],
    deps = [
        ":github_repo",
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:pbutil",
        "//labm8:test",
    ],
)

py_binary(
    name = "importer",
    srcs = ["importer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":contentfiles",
        "//datasets/github/scrape_repos/preprocessors",
        "//datasets/github/scrape_repos/preprocessors:public",
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:pbutil",
        "//third_party/py/progressbar",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "importer_test",
    srcs = ["importer_test.py"],
    deps = [
        ":contentfiles",
        ":importer",
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//labm8:app",
        "//labm8:pbutil",
        "//labm8:test",
    ],
)

py_binary(
    name = "indexer",
    srcs = ["indexer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":github_repo",
        "//datasets/github/scrape_repos/preprocessors",
        "//datasets/github/scrape_repos/preprocessors:public",
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:pbutil",
        "//third_party/py/progressbar",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "indexer_test",
    srcs = ["indexer_test.py"],
    deps = [
        ":github_repo",
        ":indexer",
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//labm8:app",
        "//labm8:pbutil",
        "//labm8:test",
    ],
)

py_binary(
    name = "scraper",
    srcs = ["scraper.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//datasets/github:api",
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:humanize",
        "//labm8:labdate",
        "//labm8:pbutil",
        "//third_party/py/github",
        "//third_party/py/progressbar",
    ],
)

py_test(
    name = "scraper_test",
    srcs = ["scraper_test.py"],
    deps = [
        ":scraper",
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//labm8:app",
        "//labm8:labdate",
        "//labm8:test",
    ],
)
