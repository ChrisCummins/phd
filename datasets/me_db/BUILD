# me.db - Aggregate personal data from several sources.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")
load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")

licenses(["restricted"])  # MIT

exports_files(["LICENSE"])

py_library(
    name = "importers",
    srcs = ["importers.py"],
    visibility = ["//datasets/me_db:__subpackages__"],
    deps = [
        ":me_proto_pb2",
        "//labm8:ppar",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "me_db",
    srcs = ["me_db.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//datasets/me_db/providers/health_kit",
        "//datasets/me_db/providers/life_cycle",
        "//datasets/me_db/providers/timing",
        "//datasets/me_db/providers/ynab",
        "//labm8:humanize",
        "//labm8:labdate",
        "//labm8:sqlutil",
        "//third_party/py/absl",
        "//third_party/py/sqlalchemy",
    ],
)

py_proto_library(
    name = "me_proto_pb2",
    protos = ["me.proto"],
    visibility = ["//datasets/me_db:__subpackages__"],
    with_grpc = False,
)

cc_proto_library(
    name = "me_proto_cc",
    protos = ["me.proto"],
    visibility = ["//datasets/me_db:__subpackages__"],
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
