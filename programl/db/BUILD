cc_library(
    name = "db",
    srcs = ["db.cc"],
    hdrs = ["db.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/cpp:logging",
        "//labm8/cpp:status",
        "//labm8/cpp:statusor",
        "//third_party/libpqxx",
    ],
)

cc_library(
    name = "directory_importer",
    srcs = ["directory_importer.cc"],
    hdrs = ["directory_importer.h"],
    visibility = ["//programl:__subpackages__"],
    deps = [
        ":db",
        ":file_checksums",
        ":text_column",
        "//labm8/cpp:logging",
        "//labm8/cpp:status",
        "//labm8/cpp:status_macros",
        "//labm8/cpp:statusor",
        "//third_party/libpqxx",
        "@boost//:filesystem",
    ],
)

cc_library(
    name = "file_checksums",
    srcs = ["file_checksums.cc"],
    hdrs = ["file_checksums.h"],
    visibility = ["//programl:__subpackages__"],
    deps = [
        "//labm8/cpp:status",
        "//labm8/cpp:statusor",
        "@boost//:filesystem",
    ],
)

cc_library(
    name = "text_column",
    srcs = ["text_column.cc"],
    hdrs = ["text_column.h"],
    visibility = ["//programl:__subpackages__"],
    deps = [
        "//labm8/cpp:string",
    ],
)
