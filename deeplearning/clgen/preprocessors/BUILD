# This package defines the preprocessor pipeline for CLgen.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_library(
    name = "clang",
    srcs = ["clang.py"],
    data = select({
        "//:darwin": [
            "@libcxx_mac//:headers",
            "@llvm_mac//:clang",
        ],
        "//conditions:default": [
            "@libcxx_linux//:headers",
            "@llvm_linux//:clang",
        ],
    }),
    deps = [
        ":public",
        "//compilers/llvm",
        "//compilers/llvm:clang_format",
        "//deeplearning/clgen:errors",
        "//lib/labm8:bazelutil",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "clang_test",
    srcs = ["clang_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":clang",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":public",
        "//deeplearning/clgen:errors",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "common_test",
    srcs = ["common_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":common",
        "//deeplearning/clgen:errors",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "cxx",
    srcs = ["cxx.py"],
    data = select({
        "//:darwin": [
            "@libcxx_mac//:headers",
        ],
        "//conditions:default": [
            "@libcxx_linux//:headers",
        ],
    }),
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":clang",
        ":normalizer",
        ":public",
        "//compilers/llvm:clang",
        "//lib/labm8:bazelutil",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "cxx_test",
    srcs = ["cxx_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cxx",
        "//deeplearning/clgen:errors",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py3_image(
    name = "cxx_test_image",
    srcs = ["cxx_test.py"],
    base = "@python3.6//image",
    main = "cxx_test.py",
    deps = [
        ":cxx",
        "//deeplearning/clgen:errors",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "java",
    srcs = ["java.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":clang",
        ":public",
        "//deeplearning/clgen:errors",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "java_test",
    srcs = ["java_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":java",
        "//deeplearning/clgen:errors",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "normalizer",
    srcs = ["normalizer.py"],
    data = [
        "//deeplearning/clgen/native:clgen-rewriter",
    ],
    deps = [
        "//deeplearning/clgen:errors",
        "//lib/labm8:bazelutil",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "normalizer_test",
    srcs = ["normalizer_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":normalizer",
        "//deeplearning/clgen:errors",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "opencl",
    srcs = ["opencl.py"],
    data = [
        "//deeplearning/clgen/data/include:opencl",
        "//deeplearning/clgen/data/include:opencl-shim",
        "//third_party/libclc/generic/include:headers",
    ],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":clang",
        ":normalizer",
        ":public",
        "//lib/labm8:bazelutil",
    ],
)

py_test(
    name = "opencl_test",
    srcs = ["opencl_test.py"],
    data = ["//deeplearning/clgen/data/include:opencl-shim"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":clang",
        ":opencl",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//lib/labm8:bazelutil",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "preprocessors",
    srcs = ["preprocessors.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        # This target *must* depend on all modules which expose a
        # @public.clgen_preprocessor decorated function, since the method
        # GetPreprocessorFunction() will try to import it. Otherwise it would
        # be the responsibility of any code which calls into one of the public
        # PreprocessX() functions to ensure that the module which implements
        # each preprocessor is listed in its deps.
        ":common",
        ":cxx",
        ":java",
        ":normalizer",
        ":opencl",
        ":public",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/proto:internal_py_pb2",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "preprocessors_test",
    srcs = ["preprocessors_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":preprocessors",
        ":public",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen:errors",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "public",
    srcs = ["public.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/absl",
    ],
)

py_test(
    name = "public_test",
    srcs = ["public_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":public",
        "//deeplearning/clgen:errors",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
