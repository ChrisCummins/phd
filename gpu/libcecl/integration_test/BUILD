filegroup(
    name = "hello_sources",
    srcs = ["hello.cc"],
)

cc_binary(
    name = "hello",
    srcs = [":hello_sources"],
    linkopts = [
        "-ldl",
        "-lpthread",
    ] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    deps = [
        "//third_party/opencl",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

py_test(
    name = "integration_test",
    srcs = ["integration_test.py"],
    data = [
        ":hello_sources",
        "//gpu/libcecl:libcecl.so",
    ],
    deps = [
        "//compilers/llvm:clang",
        "//gpu/libcecl:libcecl_rewriter",
        "//gpu/libcecl:libcecl_runtime",
        "//labm8:app",
        "//labm8:test",
    ],
)
