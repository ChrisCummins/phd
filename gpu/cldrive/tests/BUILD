py_library(
    name = "lib",
    srcs = ["lib.py"],
    deps = [
        "//gpu/cldrive",
    ],
)

py_test(
    size = "small",
    name = "args_test",
    srcs = ["args_test.py"],
    deps = [
        ":lib",
        "//gpu/cldrive",
    ],
)

py_test(
    name = "data_test",
    size = "small",
    srcs = ["data_test.py"],
    deps = [
        ":lib",
        "//gpu/cldrive",
    ],
)


py_test(
    name = "driver_test",
    size = "small",
    srcs = ["driver_test.py"],
    deps = [
        ":lib",
        "//gpu/cldrive",
    ],
)

py_test(
    name = "env_test",
    srcs = ["env_test.py"],
    size = "small",
    deps = [
        ":lib",
        "//gpu/cldrive",
    ],
)
