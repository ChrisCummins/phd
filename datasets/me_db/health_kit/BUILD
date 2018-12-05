# Package to handle data from HealthKit.

py_binary(
    name = "health_kit",
    srcs = ["health_kit.py"],
    data = [":xml_export_worker"],
    visibility = ["//datasets/me_db:__pkg__"],
    deps = [
        "//datasets/me_db:importers",
        "//datasets/me_db:me_proto_pb2",
        "//lib/labm8:bazelutil",
        "//lib/labm8:pbutil",
        "//third_party/py/absl",
    ],
)

cc_binary(
    name = "xml_export_worker",
    srcs = ["xml_export_worker.cc"],
    deps = [
        "//datasets/me_db:me_proto_cc",
        "//phd:macros",
        "//phd:pbutil",
        "//phd:string",
        "@boost//:filesystem",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)
