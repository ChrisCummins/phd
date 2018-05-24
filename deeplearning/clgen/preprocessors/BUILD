# This package defines the preprocessor pipeline for CLgen.

py_library(
    name = "clang",
    srcs = ["clang.py"],
    deps = [
        ":public",
        "//config:getconfig",
        "//deeplearning/clgen:dbutil",
        "//deeplearning/clgen:errors",
    ],
)

py_test(
    name = "clang_test",
    srcs = ["clang_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":clang",
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":public",
        "//config:getconfig",
        "//deeplearning/clgen:errors",
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
    ],
)

py_library(
    name = "cxx",
    srcs = ["cxx.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":clang",
        ":normalizer",
        ":public",
        "//deeplearning/clgen:native",
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
    ],
)

py_library(
    name = "java",
    srcs = ["java.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":clang",
        ":public",
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
    ],
)

py_library(
    name = "normalizer",
    srcs = ["normalizer.py"],
    data = ["//deeplearning/clgen/native:clgen-rewriter"],
    deps = [
        "//config:getconfig",
        "//deeplearning/clgen:errors",
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
    ],
)

py_library(
    name = "opencl",
    srcs = ["opencl.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":clang",
        ":normalizer",
        ":public",
        "//config:getconfig",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen:native",
    ],
)

py_test(
    name = "opencl_test",
    srcs = ["opencl_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":clang",
        ":opencl",
        "//deeplearning/clgen:native",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//deeplearning/clgen/tests:testlib",
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
        "//deeplearning/clgen:dbutil",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen:languages",
        "//deeplearning/clgen/proto:internal_py_pb2",
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
        "//deeplearning/clgen:dbutil",
        "//deeplearning/clgen:errors",
    ],
)

py_library(
    name = "public",
    srcs = ["public.py"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "public_test",
    srcs = ["public_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":public",
        "//deeplearning/clgen:errors",
    ],
)
