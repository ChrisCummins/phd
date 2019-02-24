# Package to handle data from YNAB (You Need A Budget).

cc_binary(
    name = "json_budget_worker",
    srcs = ["json_budget_worker.cc"],
    deps = [
        "//datasets/me_db:me_proto_cc",
        "//phd:logging",
        "//phd:pbutil",
        "//phd:string",
        "@boost//:filesystem",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

py_binary(
    name = "make_dataset",
    testonly = 1,
    srcs = ["make_dataset.py"],
    deps = [
        "//labm8:jsonutil",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "ynab",
    srcs = ["ynab.py"],
    data = [":json_budget_worker"],
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
    name = "ynab_test",
    srcs = ["ynab_test.py"],
    deps = [
        ":make_dataset",
        ":ynab",
        "//datasets/me_db:me_proto_pb2",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)
