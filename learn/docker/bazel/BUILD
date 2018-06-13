# Learning to use the bazel docker rules.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")
load("@requirements//:requirements.bzl", "requirement")

py3_image(
    name = "python_image",
    srcs = ["test.py"],
    base = "@python3.6//image",
    data = [":datafile.txt"],
    main = "test.py",
    deps = [
        "//lib/labm8:bazelutil",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)
