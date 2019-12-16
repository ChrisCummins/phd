# This module defines sequence representations for programs.

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

cc_library(
    name = "cached_string_encoder",
    srcs = ["cached_string_encoder.cc"],
    hdrs = ["cached_string_encoder.h"],
    deps = [
        "//labm8/cpp:string",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "cached_string_encoder_test",
    srcs = ["cached_string_encoder_test.cc"],
    deps = [
        ":cached_string_encoder",
        "//labm8/cpp:string",
        "//labm8/cpp:test",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

py_library(
    name = "graph2seq",
    srcs = ["graph2seq.py"],
    data = [
        ":graph_encoder_worker",
        ":llvm_vocab",
    ],
    visibility = ["//deeplearning/ml4pl/models/lstm:__subpackages__"],
    deps = [
        ":graph2seq_pb_py",
        ":ir2seq",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:humanize",
        "//labm8/py:pbutil",
        "//labm8/py:progress",
        "//third_party/py/lru_dict",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "graph2seq_test",
    size = "enormous",
    srcs = ["graph2seq_test.py"],
    deps = [
        ":graph2seq",
        ":ir2seq",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset",
        "//deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ml4pl/testing:random_graph_tuple_database_generator",
        "//deeplearning/ml4pl/testing:random_unlabelled_graph_database_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)

proto_library(
    name = "graph2seq_pb",
    srcs = ["graph2seq.proto"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl_pb",
    ],
)

python_proto_library(
    name = "graph2seq_pb_py",
    deps = [":graph2seq_pb"],
)

cc_proto_library(
    name = "graph2seq_pb_cc",
    deps = [":graph2seq_pb"],
)

cc_library(
    name = "graph_encoder",
    srcs = ["graph_encoder.cc"],
    hdrs = ["graph_encoder.h"],
    deps = [
        ":cached_string_encoder",
        ":graph2seq_pb_cc",
        ":graph_serializer",
        "//deeplearning/ml4pl/graphs:programl_pb_cc",
        "//labm8/cpp:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "graph_encoder_test",
    srcs = ["graph_encoder_test.cc"],
    deps = [
        ":graph2seq_pb_cc",
        ":graph_encoder",
        "//deeplearning/ml4pl/graphs:programl_pb_cc",
        "//labm8/cpp:test",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_binary(
    name = "graph_encoder_worker",
    srcs = ["graph_encoder_worker.cc"],
    deps = [
        ":cached_string_encoder",
        ":graph2seq_pb_cc",
        ":graph_encoder",
        "//labm8/cpp:pbutil",
    ],
)

cc_library(
    name = "graph_serializer",
    srcs = ["graph_serializer.cc"],
    hdrs = ["graph_serializer.h"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl_pb_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "graph_serializer_test",
    srcs = ["graph_serializer_test.cc"],
    deps = [
        ":graph_serializer",
        "//deeplearning/ml4pl/graphs:programl_pb_cc",
        "//labm8/cpp:test",
    ],
)

py_binary(
    name = "ir2seq",
    srcs = ["ir2seq.py"],
    data = [":llvm_vocab"],
    visibility = ["//deeplearning/ml4pl/models/lstm:__subpackages__"],
    deps = [
        ":lexers",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ncc:vocabulary",
        "//deeplearning/ncc/inst2vec:api",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:humanize",
        "//labm8/py:progress",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "ir2seq_test",
    size = "enormous",
    srcs = ["ir2seq_test.py"],
    deps = [
        ":ir2seq",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:decorators",
        "//labm8/py:test",
        "//third_party/py/numpy",
    ],
)

proto_library(
    name = "ir2seq_pb",
    srcs = ["ir2seq.proto"],
)

python_proto_library(
    name = "ir2seq_pb_py",
    deps = [":ir2seq_pb"],
)

cc_proto_library(
    name = "ir2seq_pb_cc",
    deps = [":ir2seq_pb"],
)

py_binary(
    name = "lexers",
    srcs = ["lexers.py"],
    data = [
        ":string_encoder_worker",
    ],
    deps = [
        ":ir2seq_pb_py",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:pbutil",
        "//labm8/py:progress",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "lexers_test",
    size = "enormous",
    srcs = ["lexers_test.py"],
    deps = [
        ":lexers",
        "//labm8/py:decorators",
        "//labm8/py:test",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "llvm_vocab",
    srcs = ["llvm_vocab.json"],
)

cc_binary(
    name = "string_encoder_worker",
    srcs = ["string_encoder_worker.cc"],
    deps = [
        ":cached_string_encoder",
        ":ir2seq_pb_cc",
        "//labm8/cpp:pbutil",
    ],
)
