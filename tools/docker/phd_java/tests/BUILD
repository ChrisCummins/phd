load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_test(
    name = "basic_app_test",
    srcs = ["basic_app_test.py"],
    deps = [
        "//labm8/py:app",
    ],
)

py3_image(
    name = "basic_app_test_image",
    testonly = 1,
    srcs = ["basic_app_test.py"],
    base = "@base_java//image",
    main = "basic_app_test.py",
    deps = [":basic_app_test"],
)

sh_test(
    name = "basic_app_docker_test",
    srcs = ["basic_app_test.sh"],
    data = [
        ":basic_app_test_image.tar",
    ],
)
