py_binary(
    name = "import_cgo17_clgen_1000",
    srcs = ["import_cgo17_clgen_1000.py"],
    data = ["//docs/2017_02_cgo/data:clgen_1000_tar"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//third_party/py/progressbar",
    ],
)

py_binary(
    name = "import_from_backtracking_db",
    srcs = ["import_from_backtracking_db.py"],
    deps = [
        "//deeplearning/clgen/preprocessors",
        "//experimental/deeplearning/clgen/backtracking:backtracking_db",
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//labm8/py:app",
        "//labm8/py:sqlutil",
    ],
)

py_binary(
    name = "import_from_clgen",
    srcs = ["import_from_clgen.py"],
    deps = [
        "//deeplearning/clgen",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//labm8/py:app",
        "//labm8/py:prof",
    ],
)

py_binary(
    name = "import_from_deepsmith_datastore",
    srcs = ["import_from_deepsmith_datastore.py"],
    deps = [
        "//deeplearning/deepsmith:datastore",
        "//deeplearning/deepsmith:generator",
        "//deeplearning/deepsmith:testcase",
        "//deeplearning/deepsmith:toolchain",
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//labm8/py:app",
        "//labm8/py:pbutil",
        "//labm8/py:sqlutil",
    ],
)

py_binary(
    name = "import_from_fish",
    srcs = ["import_from_fish.py"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//experimental/deeplearning/fish/proto:fish_pb_py",
        "//labm8/py:app",
        "//labm8/py:pbutil",
    ],
)

py_binary(
    name = "import_from_directory",
    srcs = ["import_from_directory.py"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//labm8/py:app",
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
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping/models:ncc",
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//labm8/py:app",
        "//third_party/py/progressbar",
    ],
)

py_binary(
    name = "import_from_legacy_clgen",
    srcs = ["import_from_legacy_clgen.py"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//labm8/py:app",
    ],
)
