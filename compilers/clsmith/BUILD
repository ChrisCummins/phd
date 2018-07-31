# A python wrapper around CLSmith, a random generator of OpenCL C programs.

load("@requirements//:requirements.bzl", "requirement")

py_binary(
    name = "clsmith",
    srcs = ["clsmith.py"],
    data = ["@CLSmith"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//lib/labm8:bazelutil",
        "//lib/labm8:fs",
        requirement("absl-py"),
    ],
)

py_test(
    name = "clsmith_test",
    srcs = ["clsmith_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":clsmith",
        requirement("absl-py"),
        requirement("pytest"),
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
        "@libopencl//:libOpenCL.so",
    ],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//gpu/cldrive:driver",
        "//gpu/cldrive:env",
        "//lib/labm8:bazelutil",
        "//lib/labm8:fs",
        "//lib/labm8:system",
        requirement("absl-py"),
    ],
)

py_test(
    name = "cl_launcher_test",
    srcs = ["cl_launcher_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cl_launcher",
        "//gpu/cldrive:env",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)
