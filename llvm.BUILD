# A package for LLVM releases.
# See: http://releases.llvm.org/download.html

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "libraries",
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
