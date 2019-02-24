# Package to handle data from Life Cycle.

# lc_export_csv_worker rule split into two, since cc_test can only depend on
# cc_binary.
cc_binary(
    name = "lc_export_csv_worker",
    srcs = ["lc_export_csv_worker.cc"],
    deps = [
        ":life_cycle_lib",
        "//phd:pbutil",
    ],
)

cc_library(
    name = "life_cycle_lib",
    srcs = ["life_cycle_lib.cc"],
    hdrs = ["life_cycle_lib.h"],
    deps = [
        "//datasets/me_db:me_proto_cc",
        "//phd:logging",
        "//phd:string",
        "@boost//:filesystem",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "life_cycle_lib_test",
    srcs = ["life_cycle_lib_test.cc"],
    deps = [
        ":life_cycle_lib",
        "//datasets/me_db:me_proto_cc",
        "//phd:test",
    ],
)

py_binary(
    name = "life_cycle",
    srcs = ["life_cycle.py"],
    data = [":lc_export_csv_worker"],
    visibility = ["//datasets/me_db:__pkg__"],
    deps = [
        "//datasets/me_db:importers",
        "//datasets/me_db:me_proto_pb2",
        "//labm8:bazelutil",
        "//labm8:pbutil",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "life_cycle_test",
    srcs = ["life_cycle_test.py"],
    deps = [
        ":life_cycle",
        ":make_dataset",
        "//datasets/me_db:me_proto_pb2",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "make_dataset",
    testonly = 1,
    srcs = ["make_dataset.py"],
    deps = [
        "//third_party/py/absl",
    ],
)

py_test(
    name = "make_dataset_test",
    srcs = ["make_dataset_test.py"],
    deps = [
        ":make_dataset",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)
