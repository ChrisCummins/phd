cc_library(
    name = "main",
    srcs = glob(
        ["src/*.cc"],
        exclude = ["src/re_posix.cc", "src/gnuregex.cc"]
    ),
    hdrs = glob(
        ["src/*.h", "include/benchmark/*.h"],
        exclude = ["src/re_posix.h", "src/gnuregex.h"]
    ),
    includes = ["include"],
    copts = ["-DHAVE_STD_REGEX"],
    visibility = ["//visibility:public"],
)
