# Scripts for graph data migrations.

py_binary(
    name = "delete_single_node_graphs",
    srcs = ["delete_single_node_graphs.py"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database",
        "//labm8:app",
    ],
)

py_binary(
    name = "migrate_graph_database",
    srcs = ["migrate_graph_database.py"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled/graph_tuple",
        "//labm8:app",
    ],
)

py_test(
    name = "migrate_graph_database_test",
    srcs = ["migrate_graph_database_test.py"],
    deps = [
        ":migrate_graph_database",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_binary(
    name = "one_hot_x_features_to_embeddings",
    srcs = ["one_hot_x_features_to_embeddings.py"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled/graph_tuple",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:prof",
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
        "//labm8:app",
        "//labm8:humanize",
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
        "//labm8:app",
        "//third_party/py/sqlalchemy",
    ],
)
