# A package for LLVM releases.

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "libs",
    srcs = glob([
        "lib/*.a",
    ]),
)

# Pre-compiled binaries, for us in data attrs of *_binary targets.

filegroup(
    name = "clang",
    srcs = ["bin/clang"],
)

filegroup(
    name = "clang-format",
    srcs = ["bin/clang-format"],
)

filegroup(
    name = "opt",
    srcs = ["bin/opt"],
)

filegroup(
    name = "libclang_so",
    srcs = ["lib/libclang.so"],
)
