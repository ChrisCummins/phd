# Labm8 is a Python library for manipulating quantitative experimental data.
#
# Each module within labm8 contains a set of related functions.

licenses(["restricted"])  # GPL v3

exports_files([
    "LICENSE",
    "README.md",
])

py_library(
    name = "archive",
    srcs = ["archive.py"],
    deps = [
        "//third_party/py/absl",
    ],
)

py_test(
    name = "archive_test",
    srcs = ["archive_test.py"],
    deps = [
        ":archive",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "bazelutil",
    srcs = ["bazelutil.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":archive",
        "//config:getconfig",
    ],
)

py_test(
    name = "bazelutil_test",
    size = "small",
    srcs = ["bazelutil_test.py"],
    data = ["data/test/hello_world"],
    deps = [
        ":bazelutil",
        ":fs",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "cache",
    srcs = ["cache.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":crypto",
        ":fs",
        ":io",
    ],
)

py_test(
    name = "cache_test",
    size = "small",
    srcs = ["cache_test.py"],
    data = ["data/test/jsoncache.json"],
    deps = [
        ":cache",
        ":fs",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "crypto",
    srcs = ["crypto.py"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "crypto_test",
    size = "small",
    srcs = ["crypto_test.py"],
    data = [
        "data/test/empty_file",
        "data/test/hello_world",
    ],
    deps = [
        ":crypto",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "graph",
    srcs = ["graph.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fmt",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "hashcache",
    srcs = ["hashcache.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":crypto",
        ":fs",
        ":sqlutil",
        "//labm8:humanize",
        "//third_party/py/absl",
        "//third_party/py/checksumdir",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "hashcache_test",
    size = "small",
    srcs = ["hashcache_test.py"],
    deps = [
        ":hashcache",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "humanize",
    srcs = ["humanize.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/humanize",
    ],
)

py_test(
    name = "humanize_test",
    size = "small",
    srcs = ["humanize_test.py"],
    deps = [
        ":humanize",
        ":labdate",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "decorators",
    srcs = ["decorators.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/absl",
    ],
)

py_test(
    name = "decorators_test",
    srcs = ["decorators_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":decorators",
        ":test",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "err",
    srcs = ["err.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "fs",
    srcs = ["fs.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":humanize",
        ":labtypes",
        "//third_party/py/send2trash",
    ],
)

py_test(
    name = "fs_test",
    size = "small",
    srcs = ["fs_test.py"],
    data = glob([
        "data/test/empty_file",
        "data/test/data1",
        "data/test/hello_world",
        "data/test/testdir/**",
    ]),
    deps = [
        ":fs",
        ":system",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "fmt",
    srcs = ["fmt.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//third_party/py/pandas",
    ],
)

py_test(
    name = "fmt_test",
    size = "small",
    srcs = ["fmt_test.py"],
    deps = [
        ":fmt",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "io",
    srcs = ["io.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":labtypes",
        ":system",
    ],
)

py_test(
    name = "io_test",
    size = "small",
    srcs = ["io_test.py"],
    deps = [
        ":io",
        ":system",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "jsonutil",
    srcs = ["jsonutil.py"],
    visibility = ["//visibility:public"],
    deps = [":fs"],
)

py_test(
    name = "jsonutil_test",
    size = "small",
    srcs = ["jsonutil_test.py"],
    deps = [
        ":fs",
        ":jsonutil",
        ":system",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "labdate",
    srcs = ["labdate.py"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "labdate_test",
    size = "small",
    srcs = ["labdate_test.py"],
    deps = [
        ":labdate",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "labmath",
    srcs = ["labmath.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":system",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
    ],
)

py_test(
    name = "labmath_test",
    size = "small",
    srcs = ["labmath_test.py"],
    deps = [
        ":labmath",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "labtypes",
    srcs = ["labtypes.py"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "labtypes_test",
    size = "small",
    srcs = ["labtypes_test.py"],
    deps = [
        ":labtypes",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "latex",
    srcs = ["latex.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":io",
        "//third_party/py/pandas",
    ],
)

py_test(
    name = "latex_test",
    size = "small",
    srcs = ["latex_test.py"],
    deps = [
        ":latex",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "lockfile",
    srcs = ["lockfile.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
        ":labdate",
        ":pbutil",
        ":system",
        "//labm8/proto:lockfile_py_pb2",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "lockfile_test",
    size = "small",
    srcs = ["lockfile_test.py"],
    deps = [
        ":lockfile",
        ":pbutil",
        ":test",
        "//labm8/proto:lockfile_py_pb2",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "logutil",
    srcs = ["logutil.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":labdate",
        "//labm8/proto:logging_py_pb2",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "logutil_test",
    size = "small",
    srcs = ["logutil_test.py"],
    deps = [
        ":logutil",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "make",
    srcs = ["make.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
        ":system",
    ],
)

py_test(
    name = "make_test",
    size = "small",
    srcs = ["make_test.py"],
    data = glob(["data/test/makeproj/**"]),
    deps = [
        ":fs",
        ":make",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "pbutil",
    srcs = ["pbutil.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":jsonutil",
        "//third_party/py/protobuf",
    ],
)

py_test(
    name = "pbutil_test",
    srcs = ["pbutil_test.py"],
    deps = [
        ":pbutil",
        ":test",
        "//labm8/proto:test_protos",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "ppar",
    srcs = ["ppar.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":bazelutil",
        ":pbutil",
        ":sqlutil",
        "//labm8:humanize",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "ppar_test",
    srcs = ["ppar_test.py"],
    data = [
        "//labm8/data/test/ppar:proto_worker",
        "//labm8/data/test/ppar:proto_worker_requires_args",
        "//labm8/data/test/ppar:unexpected_output_proto_worker",
    ],
    deps = [
        ":ppar",
        ":test",
        "//labm8/data/test/ppar:protos_py_pb2",
        "//third_party/py/absl",
        "//third_party/py/progressbar",
    ],
)

py_library(
    name = "prof",
    srcs = ["prof.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":labtypes",
        "//labm8:humanize",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "prof_test",
    size = "small",
    srcs = ["prof_test.py"],
    deps = [
        ":prof",
        ":system",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "shell",
    srcs = ["shell.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/absl",
    ],
)

py_library(
    name = "sqlutil",
    srcs = ["sqlutil.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":labdate",
        ":pbutil",
        ":text",
        "//third_party/py/absl",
        "//third_party/py/pandas",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "sqlutil_test",
    size = "small",
    srcs = ["sqlutil_test.py"],
    deps = [
        ":sqlutil",
        ":test",
        "//labm8/proto:test_protos",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "system",
    srcs = ["system.py"],
    visibility = ["//visibility:public"],
    deps = [":fs"],
)

py_test(
    name = "system_test",
    size = "small",
    srcs = ["system_test.py"],
    data = ["data/test/bin/scp"],
    deps = [
        ":fs",
        ":system",
        ":test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "tar",
    srcs = ["tar.py"],
    visibility = ["//visibility:public"],
    deps = [":fs"],
)

py_library(
    name = "test",
    testonly = 1,
    srcs = ["test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//:conftest",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_test(
    name = "test_test",
    srcs = ["test_test.py"],
    deps = [
        ":test",
    ],
)

py_library(
    name = "text",
    srcs = ["text.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
        ":system",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "text_test",
    size = "small",
    srcs = ["text_test.py"],
    deps = [
        ":test",
        ":text",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "viz",
    srcs = ["viz.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/absl",
        "//third_party/py/matplotlib",
    ],
)

py_test(
    name = "viz_test",
    size = "small",
    srcs = ["viz_test.py"],
    deps = [
        ":test",
        ":viz",
        "//third_party/py/absl",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/pytest",
    ],
)
