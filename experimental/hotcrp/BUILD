# Scripts for working with HotCRP.

py_binary(
    name = "json_to_acm_proceedings",
    srcs = ["json_to_acm_proceedings.py"],
    deps = [
        "//third_party/py/absl",
    ],
)

py_test(
    name = "json_to_acm_proceedings_test",
    srcs = ["json_to_acm_proceedings_test.py"],
    deps = [
        ":json_to_acm_proceedings",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)
