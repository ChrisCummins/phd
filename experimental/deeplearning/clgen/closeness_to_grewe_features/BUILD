# Does CLgen output programs with Grewe et al features close to benchmarks?

py_library(
    name = "grewe_features_db",
    srcs = ["grewe_features_db.py"],
    visibility = ["//experimental/deeplearning/clgen:__subpackages__"],
    deps = [
        "//gpu/portable_mapping_of_data_parallel_programs_to_opencl:feature_extractor",
        "//labm8:sqlutil",
        "//third_party/py/absl",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "import_cgo17_clgen_1000",
    srcs = ["import_cgo17_clgen_1000.py"],
    data = ["//docs/2017_02_cgo/data:clgen_1000_tar"],
    deps = [
        ":grewe_features_db",
        "//labm8:bazelutil",
        "//third_party/py/absl",
        "//third_party/py/progressbar",
    ],
)

py_binary(
    name = "import_from_directory",
    srcs = ["import_from_directory.py"],
    deps = [
        ":grewe_features_db",
        "//third_party/py/absl",
    ],
)

sh_test(
    name = "import_from_directory_smoke_test",
    srcs = ["import_from_directory_smoke_test.sh"],
    data = [":import_from_directory"],
)
