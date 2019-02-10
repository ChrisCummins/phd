# A python wrapper around CLSmith, a random generator of OpenCL C programs.

py_binary(
    name = "clsmith",
    srcs = ["clsmith.py"],
    data = ["@CLSmith"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:bazelutil",
        "//labm8:fs",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "clsmith_test",
    srcs = ["clsmith_test.py"],
    deps = [
        ":clsmith",
        "//labm8:test",
        "//third_party/py/absl",
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
        "//gpu/cldrive:driver",
        "//gpu/cldrive:env",
        "//labm8:bazelutil",
        "//labm8:fs",
        "//labm8:system",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "cl_launcher_test",
    srcs = ["cl_launcher_test.py"],
    deps = [
        ":cl_launcher",
        "//gpu/cldrive:env",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)
