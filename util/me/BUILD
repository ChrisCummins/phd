# me.db - Aggregate personal data from several sources.

licenses(["restricted"])  # MIT

exports_files(["LICENSE"])

py_library(
    name = "aggregate",
    srcs = ["aggregate.py"],
    deps = [
        ":utils",
        ":spreadsheet",
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

py_binary(
    name = "me",
    srcs = ["me.py"],
    deps = [
        ":aggregate",
        ":healthkit",
        ":omnifocus",
        ":toggl_import",
        "//third_party/py/absl",
    ],
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
