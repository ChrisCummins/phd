# Experiemnts in random optimization.

py_library(
    name = "env",
    srcs = ["env.py"],
    deps = [
        "//compilers/llvm:clang",
        "//compilers/llvm:llvm_link",
        "//compilers/llvm:opt",
        "//experimental/compilers/random_opt/proto:random_opt_py_pb2",
        "//lib/labm8:crypto",
        "//lib/labm8:labdate",
        "//third_party/py/absl",
        "//third_party/py/gym",
    ],
)

py_binary(
    name = "random_opt",
    srcs = ["random_opt.py"],
    deps = [
        ":env",
        "//compilers/llvm:opt",
        "//datasets/benchmarks:bzip2",
        "//experimental/compilers/random_opt/proto:random_opt_py_pb2",
        "//third_party/py/absl",
        "//third_party/py/gym",
    ],
)
