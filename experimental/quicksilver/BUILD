filegroup(
    name = "app_configs",
    srcs = [
        "app_a_jacobi_config.json",
        "app_b_jacobi_config.json",
    ],
)

py_binary(
    name = "jacobi_demo",
    srcs = ["jacobi_demo.py"],
    data = [
        ":app_configs",
    ],
    deps = [
        "//datasets/benchmarks/jacobi_opencl",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:humanize",
        "//labm8:jsonutil",
        "//labm8:prof",
    ],
)
