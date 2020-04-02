# Learning to use the various Boost C++ libraries.

cc_binary(
    name = "fs",
    srcs = ["fs.cc"],
    linkopts = [
        "-lpthread",
    ],
    deps = [
        "//third_party/openssl",
        "@boost//:filesystem",
        "@boost//:format",
    ],
)
