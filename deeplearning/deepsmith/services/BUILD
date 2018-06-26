# Service processes for DeepSmith.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")
load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//experimental:__subpackages__"])

py_binary(
    name = "randchar",
    srcs = ["randchar.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//deeplearning/deepsmith/cli:__subpackages__",
        "//experimental:__subpackages__",
    ],
    deps = [
        ":generator",
        ":services",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
        requirement("grpcio"),
        requirement("absl-py"),
    ],
)

py_binary(
    name = "dummy_generator",
    srcs = ["dummy_generator.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//deeplearning/deepsmith/cli:__subpackages__",
        "//experimental:__subpackages__",
    ],
    deps = [
        ":generator",
        ":services",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
        requirement("grpcio"),
        requirement("absl-py"),
    ],
)

py_library(
    name = "services",
    srcs = ["services.py"],
    visibility = [
        "//deeplearning/deepsmith/cli:__subpackages__",
        "//experimental:__subpackages__",
    ],
    deps = [
        "//deeplearning/deepsmith/proto:service_py_pb2",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
        requirement("grpcio"),
    ],
)

py_binary(
    name = "datastore",
    srcs = ["datastore.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//deeplearning/deepsmith/cli:__subpackages__",
        "//experimental:__subpackages__",
    ],
    deps = [
        ":services",
        "//deeplearning/deepsmith/proto:datastore_py_pb2",
        "//lib/labm8:pbutil",
        requirement("grpcio"),
        requirement("absl-py"),
    ],
)

py3_image(
    name = "datastore_image",
    srcs = ["datastore.py"],
    base = "@python3.6//image",
    main = "datastore.py",
    deps = [":datastore"],
)

py_library(
    name = "generator",
    srcs = ["generator.py"],
    deps = [
        ":services",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//lib/labm8:pbutil",
    ],
)

py_binary(
    name = "clgen",
    srcs = ["clgen.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//deeplearning/deepsmith/cli:__subpackages__",
        "//experimental:__subpackages__",
    ],
    deps = [
        ":generator",
        ":services",
        "//deeplearning/clgen:sample",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        requirement("grpcio"),
        requirement("absl-py"),
    ],
)

# TODO(cec): Re-enable once CLgen cyclic symlinks is fixed.
# py3_image(
#     name = "clgen_image",
#     base = "@python3.6//image",
#     srcs = ["clgen.py"],
#     deps = [":clgen"],
#     main = "clgen.py",
# )

py_library(
    name = "harness",
    srcs = ["harness.py"],
    deps = [
        ":services",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//lib/labm8:pbutil",
    ],
)

py_binary(
    name = "cldrive",
    srcs = ["cldrive.py"],
    data = ["@opencl_120_headers//:opencl_headers"] + select({
        "//:darwin": [
            "@llvm_mac//:clang",
            "@llvm_mac//:libcxx_headers",
            "@llvm_mac//:libcxx_libs",
        ],
        "//conditions:default": [
            "@llvm_linux//:clang",
            "@llvm_linux//:libcxx_headers",
            "@llvm_linux//:libcxx_libs",
            "@libopencl//:libOpenCL.so",
        ],
    }),
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//deeplearning/deepsmith/cli:__subpackages__",
        "//experimental:__subpackages__",
    ],
    deps = [
        ":harness",
        ":services",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//gpu/cldrive",
        "//gpu/oclgrind",
        "//lib/labm8:bazelutil",
        "//lib/labm8:fs",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
        "//lib/labm8:system",
        requirement("absl-py"),
        requirement("grpcio"),
    ],
)

py_test(
    name = "cldrive_test",
    srcs = ["cldrive_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cldrive",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)

py3_image(
    name = "cldrive_image",
    srcs = ["cldrive.py"],
    base = "@python3.6//image",
    main = "cldrive.py",
    deps = [":cldrive"],
)

# A docker image for the cldrive unit tests. This is useful for testing the
# hemetic-ness (definitely not a word) of the cldrive harness.
py3_image(
    name = "cldrive_test_image",
    srcs = ["cldrive_test.py"],
    base = "@python3.6//image",
    main = "cldrive_test.py",
    deps = [
        ":cldrive",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)
