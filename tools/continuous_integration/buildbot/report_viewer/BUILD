# Flask app for rendering test results.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_binary(
    name = "report_viewer",
    srcs = ["report_viewer.py"],
    data = [
        "//tools/continuous_integration/buildbot/report_viewer/static",
        "//tools/continuous_integration/buildbot/report_viewer/templates",
    ],
    deps = [
        "//:build_info",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:humanize",
        "//labm8/py:prof",
        "//third_party/py/flask",
        "//third_party/py/portpicker",
        "//third_party/py/sqlalchemy",
        "//tools/continuous_integration:bazel_test_db",
    ],
)

py3_image(
    name = "image",
    srcs = [":report_viewer"],
    base = "@base//image",
    main = "report_viewer.py",
    deps = [
        ":report_viewer",
    ],
)
