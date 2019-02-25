py_binary(
    name = "drive_with_cldrive",
    srcs = ["drive_with_cldrive.py"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//gpu/cldrive:api",
        "//gpu/cldrive/legacy:env",
        "//gpu/cldrive/proto:cldrive_py_pb2",
        "//labm8:system",
        "//labm8:text",
        "//research/cummins_2017_cgo:opencl_kernel_driver",
        "//third_party/py/absl",
    ],
)
