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
        "//:build_info",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:humanize",
        "//third_party/py/flask",
        "//third_party/py/portpicker",
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
        "//:build_info",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:humanize",
        "//third_party/py/flask",
        "//third_party/py/portpicker",
    ],
)
