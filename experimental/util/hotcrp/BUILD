# Scripts for working with HotCRP.

py_binary(
    name = "json_to_acm_proceedings",
    srcs = ["json_to_acm_proceedings.py"],
    deps = [
        "//labm8/py:app",
    ],
)

py_test(
    name = "json_to_acm_proceedings_test",
    srcs = ["json_to_acm_proceedings_test.py"],
    deps = [
        ":json_to_acm_proceedings",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)
