# Labm8 is a Python library for manipulating quantitative experimental data.
#
# Each module within labm8 contains a set of related functions.

licenses(["restricted"])  # GPL v3

exports_files([
    "LICENSE",
    "README.md",
])

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
        "//lib/labm8:crypto",
    ],
)

py_library(
    name = "labdate",
    srcs = ["labdate.py"],
)

py_test(
    name = "labdate_test",
    size = "small",
    srcs = ["labdate_test.py"],
    deps = [":labdate"],
)

py_library(
    name = "dirhashcache",
    srcs = ["dirhashcache.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
    ],
)

py_library(
    name = "fs",
    srcs = ["fs.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":labtypes",
    ],
)

py_test(
    name = "fs_test",
    size = "small",
    srcs = ["fs_test.py"],
    data = glob(["data/test/testdir/**"]),
    deps = [
        "//lib/labm8:fs",
        "//lib/labm8:system",
    ],
)

py_library(
    name = "fmt",
    srcs = ["fmt.py"],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

py_test(
    name = "fmt_test",
    size = "small",
    srcs = ["fmt_test.py"],
    deps = [":fmt"],
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

py_library(
    name = "jsonutil",
    srcs = ["jsonutil.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
    ],
)

py_library(
    name = "latex",
    srcs = ["latex.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":io",
    ],
)

py_library(
    name = "lockfile",
    srcs = ["lockfile.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
        ":system",
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

py_library(
    name = "labmath",
    srcs = ["labmath.py"],
    visibility = ["//visibility:public"],
    deps = [":system"],
)

py_library(
    name = "pbutil",
    srcs = ["pbutil.py"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "pbutil_test",
    srcs = ["pbutil_test.py"],
    deps = [
        ":pbutil",
        "//lib/labm8/proto:test_protos",
    ],
)

py_library(
    name = "prof",
    srcs = ["prof.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":labtypes",
    ],
)

py_library(
    name = "system",
    srcs = ["system.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
    ],
)

py_library(
    name = "tar",
    srcs = ["tar.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
    ],
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

py_library(
    name = "labtypes",
    srcs = ["labtypes.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "viz",
    srcs = ["viz.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":io",
    ],
)

py_library(
    name = "err",
    srcs = ["err.py"],
    visibility = ["//visibility:public"],
)

### Unit test files to refactor:

py_test(
    name = "io_test",
    size = "small",
    srcs = ["io_test.py"],
    deps = [
        "//lib/labm8:io",
        "//lib/labm8:system",
    ],
)

py_test(
    name = "jsonutil_test",
    size = "small",
    srcs = ["jsonutil_test.py"],
    deps = [
        "//lib/labm8:fs",
        "//lib/labm8:jsonutil",
        "//lib/labm8:system",
    ],
)

py_test(
    name = "latex_test",
    size = "small",
    srcs = ["latex_test.py"],
    deps = [
        "//lib/labm8:latex",
    ],
)

py_test(
    name = "lockfile_test",
    size = "small",
    srcs = ["lockfile_test.py"],
    deps = [
        "//lib/labm8:lockfile",
    ],
)

py_test(
    name = "make_test",
    size = "small",
    srcs = ["make_test.py"],
    data = glob(["data/makeproj/**"]),
    deps = [
        "//lib/labm8:fs",
        "//lib/labm8:make",
    ],
)

py_test(
    name = "labmath_test",
    size = "small",
    srcs = ["labmath_test.py"],
    deps = [
        "//lib/labm8:labmath",
    ],
)

py_test(
    name = "prof_test",
    size = "small",
    srcs = ["prof_test.py"],
    deps = [
        "//lib/labm8:prof",
        "//lib/labm8:system",
    ],
)

py_test(
    name = "system_test",
    size = "small",
    srcs = ["system_test.py"],
    deps = [
        "//lib/labm8:fs",
        "//lib/labm8:system",
    ],
)

py_test(
    name = "text_test",
    size = "small",
    srcs = ["text_test.py"],
    deps = [
        "//lib/labm8:text",
    ],
)

py_test(
    name = "labtypes_test",
    size = "small",
    srcs = ["labtypes_test.py"],
    deps = [
        "//lib/labm8:labtypes",
    ],
)

py_test(
    name = "viz_test",
    size = "small",
    srcs = ["viz_test.py"],
    deps = [
        "//lib/labm8:fs",
        "//lib/labm8:viz",
    ],
)
