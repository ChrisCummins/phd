# me.db - Aggregate personal data from several sources.

licenses(["restricted"])  # GPL v3

exports_files(["LICENSE"])

py_library(
    name = "aggregate",
    srcs = ["aggregate.py"],
    deps = [
        ":me",
        ":spreadsheet",
    ]
)

py_binary(
    name = "cli",
    srcs = ["cli.py"],
    deps = [
        ":aggregate",
        ":healthkit",
        ":omnifocus",
        ":healthkit",
        ":toggl",
    ]
)

py_library(
    name = "healthkit",
    srcs = ["healthkit.py"],
    deps = [
        ":me"
    ]
)

py_library(
    name = "me",
    srcs = ["me.py"],
)

py_library(
    name = "omnifocus",
    srcs = ["omnifocus.py"],
    deps = [
        ":me"
    ]
)

py_library(
    name = "spreadsheet",
    srcs = ["spreadsheet.py"],
)

py_library(
    name = "toggl",
    srcs = ["toggl.py"],
    deps = [
        ":me",
    ]
)
