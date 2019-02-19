py_binary(
    name = "import_cgo17_clgen_1000",
    srcs = ["import_cgo17_clgen_1000.py"],
    data = ["//docs/2017_02_cgo/data:clgen_1000_tar"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//labm8:bazelutil",
        "//third_party/py/absl",
        "//third_party/py/progressbar",
    ],
)

py_binary(
    name = "import_from_clgen",
    srcs = ["import_from_clgen.py"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//deeplearning/clgen",
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "import_from_deepsmith_datastore",
    srcs = ["import_from_deepsmith_datastore.py"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//deeplearning/deepsmith:datastore",
        "//deeplearning/deepsmith:generator",
        "//deeplearning/deepsmith:testcase",
        "//deeplearning/deepsmith:toolchain",
        "//labm8:pbutil",
        "//labm8:sqlutil",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "import_from_fish",
    srcs = ["import_from_fish.py"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//experimental/deeplearning/fish/proto:fish_py_pb2",
        "//labm8:pbutil",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "import_from_directory",
    srcs = ["import_from_directory.py"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//third_party/py/absl",
    ],
)

sh_test(
    name = "import_from_directory_smoke_test",
    srcs = ["import_from_directory_smoke_test.sh"],
    data = [":import_from_directory"],
)

py_binary(
    name = "import_from_gpgpu_benchmarks",
    srcs = ["import_from_gpgpu_benchmarks.py"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping/models:ncc",
        "//third_party/py/absl",
        "//third_party/py/progressbar",
    ],
)

py_binary(
    name = "import_from_legacy_clgen",
    srcs = ["import_from_legacy_clgen.py"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//third_party/py/absl",
    ],
)
