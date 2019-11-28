# A Jacobi iterative stencil benchmark, implemented in OpenCL.

py_binary(
    name = "jacobi_opencl",
    srcs = ["jacobi_opencl.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/numpy",
        "//third_party/py/pyopencl",
    ],
)

py_test(
    name = "jacobi_opencl_test",
    srcs = ["jacobi_opencl_test.py"],
    deps = [
        ":jacobi_opencl",
        "//labm8/py:test",
    ],
)
