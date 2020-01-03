# Learning to use the various Boost C++ libraries.

cc_binary(
    name = "fs",
    srcs = ["fs.cc"],
    linkopts = [
        "-lpthread",
        "-lcrypto",
    ],
    deps = [
        "@boost//:filesystem",
        "@boost//:format",
    ],
)
