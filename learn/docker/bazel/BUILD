# Learning to use the bazel docker rules.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py3_image(
    name = "python_image",
    srcs = ["test.py"],
    base = "@base//image",
    data = [":datafile.txt"],
    main = "test.py",
    deps = [
        "//labm8:app",
        "//labm8:bazelutil",
        "//third_party/py/pytest",
    ],
)
