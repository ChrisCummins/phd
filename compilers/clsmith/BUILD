# A python wrapper around CLSmith, a random generator of OpenCL C programs.

py_binary(
    name = "clsmith",
    srcs = ["clsmith.py"],
    data = ["@CLSmith"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:fs",
    ],
)

py_test(
    name = "clsmith_test",
    srcs = ["clsmith_test.py"],
    deps = [
        ":clsmith",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_binary(
    name = "cl_launcher",
    srcs = ["cl_launcher.py"],
    data = [
        "@CLSmith//:cl_launcher",
        "@CLSmith//:cl_safe_math_macros",
        "@CLSmith//:runtime_headers",
        "@CLSmith//:safe_math_macros",
    ] + select({
        "//:darwin": [],
        "//conditions:default": [
            "@libopencl//:libOpenCL.so",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//gpu/cldrive/legacy:driver",
        "//gpu/cldrive/legacy:env",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:fs",
        "//labm8:system",
    ],
)

py_test(
    name = "cl_launcher_test",
    srcs = ["cl_launcher_test.py"],
    deps = [
        ":cl_launcher",
        "//gpu/cldrive/legacy:env",
        "//labm8:app",
        "//labm8:test",
    ],
)
