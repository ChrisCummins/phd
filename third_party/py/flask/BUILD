# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "flask",
    srcs = ["flask.py"],
    deps = [
        "//third_party/py/requests",
        requirement("Flask"),
        requirement("Flask-Compress"),
        requirement("flask-cors"),
        requirement("Werkzeug"),
        requirement("jinja2"),
        requirement("Click"),
        requirement("itsdangerous"),
        requirement("urllib3"),
        requirement("chardet"),
        requirement("certifi"),
        requirement("retrying"),
    ],
)
