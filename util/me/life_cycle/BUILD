# Package to handle data from Life Cycle.

cc_binary(
    name = "lc_export_csv_worker",
    srcs = ["lc_export_csv_worker.cc"],
    deps = [
        "//phd:macros",
        "//phd:pbutil",
        "//phd:string",
        "//util/me:me_proto_cc",
        "@boost//:filesystem",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

py_binary(
    name = "life_cycle",
    srcs = ["life_cycle.py"],
    data = [":lc_export_csv_worker"],
    visibility = ["//util/me:__pkg__"],
    deps = [
        "//lib/labm8:bazelutil",
        "//lib/labm8:pbutil",
        "//third_party/py/absl",
        "//util/me:importers",
        "//util/me:me_proto_pb2",
    ],
)

py_test(
    name = "life_cycle_test",
    srcs = ["life_cycle_test.py"],
    deps = [
        ":life_cycle",
        ":make_dataset",
        "//third_party/py/absl",
        "//third_party/py/pytest",
        "//util/me:me_proto_pb2",
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
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
