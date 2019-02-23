# Package to handle data from HealthKit.

py_binary(
    name = "health_kit",
    srcs = ["health_kit.py"],
    data = [":xml_export_worker"],
    visibility = ["//datasets/me_db:__pkg__"],
    deps = [
        "//datasets/me_db:importers",
        "//datasets/me_db:me_proto_pb2",
        "//labm8:bazelutil",
        "//labm8:pbutil",
        "//third_party/py/absl",
    ],
)

cc_library(
    name = "health_kit_lib",
    srcs = ["health_kit_lib.cc"],
    hdrs = ["health_kit_lib.h"],
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
    name = "health_kit_lib_test",
    srcs = ["health_kit_lib_test.cc"],
    deps = [
        ":health_kit_lib",
        "//datasets/me_db:me_proto_cc",
        "//phd:test",
    ],
)

cc_binary(
    name = "xml_export_worker",
    srcs = ["xml_export_worker.cc"],
    deps = [
        ":health_kit_lib",
        "//phd:pbutil",
    ],
)
