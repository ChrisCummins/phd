# Labm8 is a Python library for manipulating quantitative experimental data.
#
# Each module within labm8 contains a set of related functions.

licenses(["restricted"])  # GPL v3

exports_files([
    "LICENSE",
    "README.md",
])

py_library(
    name = "bazelutil",
    srcs = ["bazelutil.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//config:getconfig",
    ],
)

py_test(
    name = "bazelutil_test",
    size = "small",
    srcs = ["bazelutil_test.py"],
    data = ["data/test/hello_world"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":bazelutil",
        ":fs",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cache",
        ":fs",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":crypto",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
        "//third_party/py/absl",
        "//third_party/py/checksumdir",
        "//third_party/py/humanize",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "hashcache_test",
    size = "small",
    srcs = ["hashcache_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":hashcache",
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
        ":labtypes",
        "//third_party/py/humanize",
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
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fs",
        ":system",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fmt",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":io",
        ":system",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fs",
        ":jsonutil",
        ":system",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":labdate",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":labmath",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":labtypes",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":latex",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":lockfile",
        ":pbutil",
        "//labm8/proto:lockfile_py_pb2",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":logutil",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fs",
        ":make",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

# A Jupyter notebook launcher that includes a bunch of dependencies which I
# find useful. If you would like to include extra dependencies, create a new
# py_binary target with this file as the srcs, and add the dependencies, like
# this:
#
#    py_binary(
#      name = "notebooks",
#      srcs = ["//labm8:notebooks"],
#      deps = [
#          # Your dependencies here.
#      ],
#    )
py_binary(
    name = "notebooks",
    srcs = ["notebooks.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":bazelutil",
        ":pbutil",
        ":sqlutil",
        ":viz",
        "//third_party/py/absl",
        "//third_party/py/bokeh",
        "//third_party/py/jupyter",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/seaborn",
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
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":pbutil",
        "//labm8/proto:test_protos",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "ppar",
    srcs = ["ppar.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":bazelutil",
        ":pbutil",
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
        "//labm8/data/test/ppar:protos_py_pb2",
        "//third_party/py/absl",
        "//third_party/py/progressbar",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "prof",
    srcs = ["prof.py"],
    visibility = ["//visibility:public"],
    deps = [":labtypes"],
)

py_test(
    name = "prof_test",
    size = "small",
    srcs = ["prof_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":prof",
        ":system",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
        ":pbutil",
        "//third_party/py/absl",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "sqlutil_test",
    size = "small",
    srcs = ["sqlutil_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sqlutil",
        "//labm8/proto:test_protos",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fs",
        ":system",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "tar",
    srcs = ["tar.py"],
    visibility = ["//visibility:public"],
    deps = [":fs"],
)

py_library(
    name = "text",
    srcs = ["text.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
        ":system",
    ],
)

py_test(
    name = "text_test",
    size = "small",
    srcs = ["text_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":text",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "viz",
    srcs = ["viz.py"],
    visibility = ["//visibility:public"],
    deps = [":io"],
)

py_test(
    name = "viz_test",
    size = "small",
    srcs = ["viz_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fs",
        ":viz",
        "//third_party/py/absl",
        "//third_party/py/matplotlib",
        "//third_party/py/pytest",
    ],
)
