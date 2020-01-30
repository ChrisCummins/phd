# Flask app for serving a dashboard over HTTP.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_library(
    name = "dashboard",
    srcs = ["dashboard.py"],
    data = [
        "//deeplearning/clgen/dashboard/static",
        "//deeplearning/clgen/dashboard/templates",
    ],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//labm8/py:labdate",
        "//labm8/py:sqlutil",
        "//third_party/py/flask",
        "//third_party/py/portpicker",
        "//third_party/py/sqlalchemy",
    ],
)

py_library(
    name = "dashboard_db",
    srcs = ["dashboard_db.py"],
    data = [
        "//deeplearning/clgen/dashboard/static",
        "//deeplearning/clgen/dashboard/templates",
    ],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//labm8/py:labdate",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)
