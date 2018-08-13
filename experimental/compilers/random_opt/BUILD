# Experiemnts in random optimization.

py_library(
    name = "implementation",
    srcs = ["implementation.py"],
    deps = [
        "//compilers/llvm",
        "//compilers/llvm:clang",
        "//compilers/llvm:llvm_link",
        "//compilers/llvm:opt",
        "//experimental/compilers/random_opt/proto:random_opt_py_pb2",
        "//lib/labm8:crypto",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
        "//lib/labm8:text",
        "//third_party/py/absl",
        "//third_party/py/gym",
    ],
)

py_test(
    name = "implementation_test",
    srcs = ["implementation_test.py"],
    deps = [
        ":implementation",
        "//compilers/llvm:clang",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "environments",
    srcs = ["environments.py"],
    deps = [
        # Note that :implementation is not directly imported, but required by
        # the gym registration process.
        ":implementation",
        "//compilers/llvm:opt",
        "//third_party/py/absl",
        "//datasets/benchmarks:bzip2",
        "//datasets/benchmarks:llvm_test_suite",
        "//experimental/compilers/random_opt/proto:random_opt_py_pb2",
    ],
)

py_test(
    name = "environments_test",
    srcs = ["environments_test.py"],
    deps = [
        ":environments",
        "//third_party/py/absl",
        "//third_party/py/gym",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "notebooks",
    srcs = ["notebooks.py"],
    deps = [
        "//datasets/github/scrape_repos:contentfiles",
        "//deeplearning/clgen",
        "//experimental/deeplearning/polyglot:get_instances",
        "//lib/labm8:bazelutil",
        "//lib/labm8:pbutil",
        "//lib/labm8:viz",
        "//third_party/py/absl",
        "//third_party/py/jupyter",
    ],
)

py_binary(
    name = "random_opt",
    srcs = ["random_opt.py"],
    deps = [
        ":environments",
        ":implementation",
        "//experimental/compilers/random_opt/proto:random_opt_py_pb2",
        "//lib/labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/gym",
    ],
)
