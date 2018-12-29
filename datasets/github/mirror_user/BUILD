# Mirror a GitHub user's repositories.

licenses(["notice"])  # MIT

exports_files(["LICENSE"])

py_binary(
    name = "mirror_user",
    srcs = ["mirror_user.py"],
    deps = [
        "//datasets/github:non_hemetic_credentials_file",
        "//labm8:jsonutil",
        "//third_party/py/github",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "mirror_user_test",
    srcs = ["mirror_user_test.py"],
    deps = [
        "//datasets/github:non_hemetic_credentials_file",
        ":mirror_user",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ]
)
