# Libraries for working with GitHub.

py_library(
    name = "non_hemetic_credentials_file",
    srcs = ["non_hemetic_credentials_file.py"],
    visibility = ["//datasets/github:__subpackages__"],
    deps = [
        "//third_party/py/absl",
        "//third_party/py/github",
    ],
)

py_test(
    name = "non_hemetic_credentials_file_test",
    srcs = ["non_hemetic_credentials_file_test.py"],
    deps = [
        ":non_hemetic_credentials_file",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
