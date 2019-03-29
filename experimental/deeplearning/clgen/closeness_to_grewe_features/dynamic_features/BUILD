# Drive programs to get dynamic features.

py_library(
    name = "conftest",
    testonly = 1,
    srcs = ["conftest.py"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//gpu/cldrive/legacy:env",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "drive_with_cldrive",
    srcs = ["drive_with_cldrive.py"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//gpu/cldrive:api",
        "//gpu/cldrive/legacy:env",
        "//gpu/cldrive/proto:cldrive_py_pb2",
        "//labm8:app",
        "//labm8:prof",
        "//labm8:system",
        "//labm8:text",
        "//research/cummins_2017_cgo:opencl_kernel_driver",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "drive_with_cldrive_test",
    srcs = ["drive_with_cldrive_test.py"],
    deps = [
        ":conftest",
        ":drive_with_cldrive",
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//gpu/cldrive/legacy:env",
        "//gpu/cldrive/proto:cldrive_py_pb2",
        "//labm8:system",
        "//labm8:test",
        "//third_party/py/numpy",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "import_from_legacy_gpgpu",
    srcs = ["import_from_legacy_gpgpu.py"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//gpu/libcecl:libcecl_runtime",
        "//gpu/libcecl/proto:libcecl_py_pb2",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:prof",
    ],
)

py_test(
    name = "import_from_legacy_gpgpu_test",
    srcs = ["import_from_legacy_gpgpu_test.py"],
    data = ["//experimental/deeplearning/clgen/closeness_to_grewe_features/dynamic_features/tests/data:legacy_gpgpu_logs"],
    deps = [
        ":import_from_legacy_gpgpu",
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "run_gpgpu_benchmarks",
    srcs = ["run_gpgpu_benchmarks.py"],
    deps = [
        "//datasets/benchmarks/gpgpu",
        "//datasets/benchmarks/gpgpu:gpgpu_py_pb2",
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//gpu/cldrive/legacy:env",
        "//labm8:app",
        "//labm8:sqlutil",
    ],
)

py_test(
    name = "run_gpgpu_benchmarks_test",
    srcs = ["run_gpgpu_benchmarks_test.py"],
    deps = [
        ":conftest",
        ":run_gpgpu_benchmarks",
        "//datasets/benchmarks/gpgpu:gpgpu_py_pb2",
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//gpu/clinfo/proto:clinfo_py_pb2",
        "//gpu/libcecl/proto:libcecl_py_pb2",
        "//labm8:test",
    ],
)
