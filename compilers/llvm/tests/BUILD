# Additional tests for LLVM.

py_test(
    name = "pipeline_test",
    srcs = ["pipeline_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//compilers/llvm:clang",
        "//compilers/llvm:opt",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)
