cc_library(
    name = "main",
    srcs = glob(["src/*.cc"]),
    hdrs = glob(["include/rt/*.h"]),
    includes = ["include"],
    visibility = ["//experimental/graphics/rt:__subpackages__"],
    deps = ["//third_party/tbb"],
)
