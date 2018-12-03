# Package to handle data from HealthKit.

py_binary(
    name = "healthkit",
    srcs = ["healthkit.py"],
    data = [":xml_export_worker"],
    visibility = ["//util/me:__pkg__"],
    deps = [
        "//lib/labm8:bazelutil",
        "//lib/labm8:pbutil",
        "//third_party/py/absl",
        "//util/me:importers",
        "//util/me:me_proto_pb2",
    ],
)

cc_binary(
    name = "xml_export_worker",
    srcs = ["xml_export_worker.cc"],
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
