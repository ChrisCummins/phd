py_library(
    name = "lib",
    srcs = ["lib.py"],
    deps = [
        "//gpu/cldrive",
    ],
)

py_test(
    name = "args_test",
    size = "small",
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
    size = "small",
    srcs = ["env_test.py"],
    deps = [
        ":lib",
        "//gpu/cldrive",
    ],
)
