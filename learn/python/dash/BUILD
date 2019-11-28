# Learning to use Dash.

py_binary(
    name = "hello_dash",
    srcs = ["hello_dash.py"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/dash",
        "//third_party/py/pandas",
    ],
)

py_test(
    name = "hello_dash_test",
    srcs = ["hello_dash_test.py"],
    deps = [
        ":hello_dash",
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//labm8/py:test",
        "//third_party/py/dash",
        "//third_party/py/portpicker",
    ],
)
