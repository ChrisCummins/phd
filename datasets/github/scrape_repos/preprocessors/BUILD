# This package defines contentfile preprocessors.
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

py_library(
    name = "conftest",
    testonly = 1,
    srcs = ["conftest.py"],
    deps = [
        "//labm8:test",
    ],
)

py_library(
    name = "extractors",
    srcs = ["extractors.py"],
    data = [
        ":JavaMethodsBatchedExtractor",
        ":JavaMethodsExtractor",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":public",
        "//datasets/github/scrape_repos/proto:scrape_repos_pb_py",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:pbutil",
    ],
)

py_test(
    name = "extractors_test",
    size = "small",
    srcs = ["extractors_test.py"],
    deps = [
        ":conftest",
        ":extractors",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "inliners",
    srcs = ["inliners.py"],
    data = select({
        "//:darwin": [
            "@libcxx_mac//:headers",
        ],
        "//conditions:default": [
            "@libcxx_linux//:headers",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":public",
        "//labm8:app",
        "//labm8:bazelutil",
        "//third_party/py/fuzzywuzzy",
    ],
)

py_test(
    name = "inliners_test",
    size = "small",
    srcs = ["inliners_test.py"],
    deps = [
        ":conftest",
        ":inliners",
        "//labm8:app",
        "//labm8:test",
    ],
)

java_binary(
    name = "JavaMethodsExtractor",
    srcs = ["JavaMethodsExtractor.java"],
    main_class = "datasets.github.scrape_repos.preprocessors.JavaMethodsExtractor",
    deps = [
        "//datasets/github/scrape_repos/proto:scrape_repos_pb_java",
        "@com_google_guava//jar",
        "@com_google_protobuf_java//jar",
        "@org_eclipse_core_contenttype//jar",
        "@org_eclipse_core_runtime//jar",
        "@org_eclipse_equinox_common//jar",
        "@org_eclipse_equinox_preferences//jar",
        "@org_eclipse_jdt_core//jar",
        "@org_eclipse_jface//jar",
        "@org_eclipse_jobs//jar",
        "@org_eclipse_osgi//jar",
        "@org_eclipse_osgi_util//jar",
        "@org_eclipse_resources//jar",
        "@org_eclipse_text//jar",
        "@org_osgi_framework//jar",
        "@org_osgi_service_prefs//jar",
    ],
)

java_binary(
    name = "JavaMethodsBatchedExtractor",
    srcs = ["JavaMethodsBatchedExtractor.java"],
    main_class = "datasets.github.scrape_repos.preprocessors.JavaMethodsBatchedExtractor",
    deps = [
        "//datasets/github/scrape_repos/proto:scrape_repos_pb_java",
        "@com_google_guava//jar",
        "@com_google_protobuf_java//jar",
        "@org_eclipse_core_contenttype//jar",
        "@org_eclipse_core_runtime//jar",
        "@org_eclipse_equinox_common//jar",
        "@org_eclipse_equinox_preferences//jar",
        "@org_eclipse_jdt_core//jar",
        "@org_eclipse_jface//jar",
        "@org_eclipse_jobs//jar",
        "@org_eclipse_osgi//jar",
        "@org_eclipse_osgi_util//jar",
        "@org_eclipse_resources//jar",
        "@org_eclipse_text//jar",
        "@org_osgi_framework//jar",
        "@org_osgi_service_prefs//jar",
    ],
)

py_library(
    name = "preprocessors",
    srcs = ["preprocessors.py"],
    visibility = ["//visibility:public"],
    deps = [
        # This target should depend on every target defining preprocessors, so
        # that the runfiles can be resolved.
        ":extractors",
        ":inliners",
        ":public",
        ":secrets",
        "//labm8:app",
    ],
)

py_test(
    name = "preprocessors_test",
    size = "small",
    srcs = ["preprocessors_test.py"],
    deps = [
        ":conftest",
        ":preprocessors",
        ":public",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "public",
    srcs = ["public.py"],
    visibility = ["//datasets/github/scrape_repos:__subpackages__"],
    deps = [
        "//labm8:app",
        "//labm8:fs",
    ],
)

py_test(
    name = "public_test",
    size = "small",
    srcs = ["public_test.py"],
    deps = [
        ":conftest",
        ":public",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "secrets",
    srcs = ["secrets.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":public",
        "//labm8:app",
        "//third_party/py/detect_secrets",
    ],
)

py_test(
    name = "secrets_test",
    srcs = ["secrets_test.py"],
    deps = [
        ":secrets",
        "//labm8:test",
    ],
)
