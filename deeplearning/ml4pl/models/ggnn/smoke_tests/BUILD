py_library(
    name = "smoke_test_base",
    srcs = ["smoke_test_base.py"],
    deps = [
        "//labm8:app",
        "//deeplearning/ml4pl/graphs:graph_database",
    ]
)

py_test(
    name = "ggnn_graph_classifier_smoke_test",
    srcs = ["ggnn_graph_classifier_smoke_test.py"],
    deps = [
        ":smoke_test_base",
        "//deeplearning/ml4pl/models/ggnn:ggnn_graph_classifier",
        "//labm8:app",
        "//deeplearning/ml4pl/graphs:graph_database",
    ]
)

py_test(
    name = "ggnn_node_classifier_smoke_test",
    srcs = ["ggnn_node_classifier_smoke_test.py"],
    deps = [
        ":smoke_test_base",
        "//deeplearning/ml4pl/models/ggnn:ggnn_node_classifier",
        "//labm8:app",
        "//deeplearning/ml4pl/graphs:graph_database",
    ]
)
