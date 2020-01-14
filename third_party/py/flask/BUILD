# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "flask",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        "//third_party/py/requests",
        requirement("Flask"),
        requirement("Flask-Compress"),
        requirement("flask-cors"),
        requirement("Flask-SQLalchemy"),
        requirement("Werkzeug"),
        requirement("jinja2"),
        "//third_party/py/click",
        requirement("itsdangerous"),
        requirement("chardet"),
        requirement("certifi"),
        requirement("retrying"),
    ],
)
