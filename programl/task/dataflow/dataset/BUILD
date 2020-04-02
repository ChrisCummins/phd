py_binary(
    name = "create",
    srcs = ["create.py"],
    data = ["create_labels"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:labtypes",
        "//labm8/py:pbutil",
        "//labm8/py:progress",
        "//programl/ir/llvm:inst2vec_encoder",
        "//programl/ir/llvm/py:llvm",
        "//programl/proto:ir_py",
        "//third_party/py/mysql",
    ],
)

cc_binary(
    name = "create_labels",
    srcs = ["create_labels.cc"],
    deps = [
        "//labm8/cpp:app",
        "//labm8/cpp:logging",
        "//programl/graph/analysis:reachability",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
        "//third_party/tbb",
        "@boost//:filesystem",
        "@com_google_absl//absl/strings:str_format",
    ],
)
