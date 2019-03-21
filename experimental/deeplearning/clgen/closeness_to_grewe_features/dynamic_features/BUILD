# Drive programs to get dynamic features.

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
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "drive_with_cldrive_test",
    srcs = ["drive_with_cldrive_test.py"],
    deps = [
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
