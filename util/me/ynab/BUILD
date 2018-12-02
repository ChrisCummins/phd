# Package to handle data from YNAB (You Need A Budget).

cc_binary(
    name = "json_budget_worker",
    srcs = ["json_budget_worker.cc"],
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
    name = "make_dataset",
    testonly = 1,
    srcs = ["make_dataset.py"],
    deps = [
        "//lib/labm8:jsonutil",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "ynab",
    srcs = ["ynab.py"],
    data = [":json_budget_worker"],
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
    name = "ynab_test",
    srcs = ["ynab_test.py"],
    deps = [
        ":make_dataset",
        ":ynab",
        "//third_party/py/absl",
        "//third_party/py/pytest",
        "//util/me:me_proto_pb2",
    ],
)
