cc_binary(
    name = "db-llvm2graph",
    srcs = ["db-llvm2graph.cc"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:clang++",
        ],
        "//conditions:default": [
            "@llvm_linux//:clang++",
        ],
    }),
    linkopts = [
        "-lpqxx",
        "-lpq",
    ],
    deps = [
        "//labm8/cpp:app",
        "//labm8/cpp:bazelutil",
        "//labm8/cpp:logging",
        "//labm8/cpp:statusor",
        "//programl/proto:program_graph_options_cc",
        "//third_party/libpqxx",
        "@subprocess",
    ],
)
