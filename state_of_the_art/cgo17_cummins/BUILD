py_library(
    name = "cgo17_driver",
    srcs = ["cgo17_driver.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/absl",
        "//compilers/llvm:clang",
        "//deeplearning/deepsmith/harnesses:cldrive",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//gpu/cldrive:env",
        "//gpu/libcecl/proto:libcecl_py_pb2",
        "//gpu/libcecl:libcecl_rewriter",
        "//gpu/libcecl:libcecl_compile",
        "//gpu/libcecl:libcecl_runtime",
    ],
)
