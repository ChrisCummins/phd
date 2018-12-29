# Mirror a GitHub user's repositories.

licenses(["notice"])  # MIT

exports_files(["LICENSE"])

py_binary(
    name = "mirror_user",
    srcs = ["mirror_user.py"],
    deps = [
        "//datasets/github:non_hemetic_credentials_file",
        "//labm8:jsonutil",
        "//third_party/py/absl",
        "//third_party/py/github",
    ],
)

py_test(
    name = "mirror_user_test",
    srcs = ["mirror_user_test.py"],
    deps = [
        ":mirror_user",
        "//datasets/github:non_hemetic_credentials_file",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
