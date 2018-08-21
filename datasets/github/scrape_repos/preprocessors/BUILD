# This package defines contentfile preprocessors.

py_library(
    name = "conftest",
    srcs = ["conftest.py"],
    deps = [
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "extractors",
    srcs = ["extractors.py"],
    data = [
        ":JavaMethodsExtractor",
    ],
    visibility = ["//datasets/github/scrape_repos:__subpackages__"],
    deps = [
        ":public",
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//lib/labm8:bazelutil",
        "//lib/labm8:pbutil",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "extractors_test",
    size = "small",
    srcs = ["extractors_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":extractors",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
    visibility = ["//datasets/github/scrape_repos:__subpackages__"],
    deps = [
        ":public",
        "//lib/labm8:bazelutil",
        "//third_party/py/absl",
        "//third_party/py/fuzzywuzzy",
    ],
)

py_test(
    name = "inliners_test",
    size = "small",
    srcs = ["inliners_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":inliners",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

java_binary(
    name = "JavaMethodsExtractor",
    srcs = ["JavaMethodsExtractor.java"],
    main_class = "datasets.github.scrape_repos.preprocessors.JavaMethodsExtractor",
    deps = [
        "//datasets/github/scrape_repos/proto:scrape_repos_java",
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
    visibility = ["//datasets/github/scrape_repos:__subpackages__"],
    deps = [
        ":inliners",
        ":public",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "preprocessors_test",
    size = "small",
    srcs = ["preprocessors_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":preprocessors",
        ":public",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "public",
    srcs = ["public.py"],
    visibility = ["//datasets/github/scrape_repos:__subpackages__"],
    deps = [
        "//lib/labm8:fs",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "public_test",
    size = "small",
    srcs = ["public_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":public",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
