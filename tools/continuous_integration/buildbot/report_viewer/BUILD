# Flask app for rendering test results.

py_binary(
    name = "report_viewer",
    srcs = ["report_viewer.py"],
    data = [
        "//tools/continuous_integration/buildbot/report_viewer/static",
        "//tools/continuous_integration/buildbot/report_viewer/templates",
    ],
    deps = [
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:humanize",
        "//labm8:prof",
        "//third_party/py/flask",
        "//third_party/py/portpicker",
        "//third_party/py/sqlalchemy",
        "//tools/continuous_integration:bazel_test_db",
    ],
)
