# me.db - Aggregate personal data from several sources.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")
load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")

licenses(["restricted"])  # MIT

exports_files(["LICENSE"])

py_library(
    name = "aggregate",
    srcs = ["aggregate.py"],
    deps = [
        ":spreadsheet",
        ":utils",
    ],
)

py_library(
    name = "healthkit",
    srcs = ["healthkit.py"],
    deps = [
        ":utils",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "importers",
    srcs = ["importers.py"],
    visibility = ["//util/me:__subpackages__"],
    deps = [
        ":me_proto_pb2",
        "//lib/labm8:ppar",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "me",
    srcs = ["me.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":aggregate",
        ":healthkit",
        ":omnifocus",
        ":toggl_import",
        "//lib/labm8:labdate",
        "//lib/labm8:sqlutil",
        "//third_party/py/absl",
        "//third_party/py/sqlalchemy",
        "//util/me/healthkit",
        "//util/me/life_cycle",
        "//util/me/ynab",
    ],
)

py_proto_library(
    name = "me_proto_pb2",
    protos = ["me.proto"],
    visibility = ["//util/me:__subpackages__"],
    with_grpc = False,
)

cc_proto_library(
    name = "me_proto_cc",
    protos = ["me.proto"],
    visibility = ["//util/me:__subpackages__"],
    with_grpc = False,
)

py_library(
    name = "omnifocus",
    srcs = ["omnifocus.py"],
    deps = [
        ":utils",
    ],
)

py_library(
    name = "spreadsheet",
    srcs = ["spreadsheet.py"],
    deps = [
        "//third_party/py/gspread",
        "//third_party/py/oauth2client",
    ],
)

py_library(
    name = "toggl_import",
    srcs = ["toggl_import.py"],
    deps = [
        ":utils",
        "//third_party/py/toggl",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
)
