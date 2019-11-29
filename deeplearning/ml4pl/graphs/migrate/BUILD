# Scripts for graph data migrations.

py_binary(
    name = "delete_single_node_graphs",
    srcs = ["delete_single_node_graphs.py"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database",
        "//labm8/py:app",
    ],
)

py_binary(
    name = "migrate_graph_database",
    srcs = ["migrate_graph_database.py"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled/graph_tuple",
        "//labm8/py:app",
    ],
)

py_test(
    name = "migrate_graph_database_test",
    srcs = ["migrate_graph_database_test.py"],
    deps = [
        ":migrate_graph_database",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "networkx_to_protos",
    srcs = ["networkx_to_protos.py"],
    visibility = ["//deeplearning/ml4pl/graphs/unlabelled:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:ppar",
        "//labm8/py:prof",
        "//third_party/py/networkx",
        "//third_party/py/tqdm",
    ],
)

py_test(
    name = "networkx_to_protos_test",
    srcs = ["networkx_to_protos_test.py"],
    data = [
        "//deeplearning/ml4pl/testing/data:100_unlabelled_networkx_graphs",
        "//deeplearning/ml4pl/testing/data:100_unlabelled_networkx_graphs_db",
    ],
    deps = [
        ":networkx_to_protos",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:random_cdfg_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:decorators",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)

py_binary(
    name = "one_hot_x_features_to_embeddings",
    srcs = ["one_hot_x_features_to_embeddings.py"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled/graph_tuple",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:prof",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "reapply_split",
    srcs = ["reapply_split.py"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled/graph_tuple:graph_batcher",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//third_party/py/numpy",
        "//third_party/py/scikit_learn",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "propagate_split",
    srcs = ["propagate_split.py"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database",
        "//labm8/py:app",
        "//third_party/py/sqlalchemy",
    ],
)
