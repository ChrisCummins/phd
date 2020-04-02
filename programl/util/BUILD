cc_library(
    name = "filesystem_cache",
    srcs = ["filesystem_cache.cc"],
    hdrs = ["filesystem_cache.h"],
    visibility = ["//programl:__subpackages__"],
    deps = [
        "//labm8/cpp:fsutil",
        "//labm8/cpp:logging",
        "//labm8/cpp:string",
        "@boost//:filesystem",
    ],
)

cc_library(
    name = "nproc",
    srcs = ["nproc.cc"],
    hdrs = ["nproc.h"],
    visibility = ["//programl:__subpackages__"],
    deps = [
        "@subprocess",
    ],
)
