load("@requirements//:requirements.bzl", "requirement")

licenses(["restricted"])

py_library(
    name = "db",
    srcs = ["db.py"],
    deps = [
        requirement("SQLAlchemy"),
        requirement("absl-py"),
    ],
)

py_test(
    name = "tests/db_test",
    size = "small",
    srcs = ["tests/db_test.py"],
    deps = [
        ":db",
        requirement("absl-py"),
    ],
)
