py_library(
    name = "smoke_test_base",
    srcs = ["smoke_test_base.py"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database",
        "//labm8:app",
    ],
)

py_test(
    name = "ggnn_node_classifier_smoke_test",
    size = "enormous",
    srcs = ["ggnn_node_classifier_smoke_test.py"],
    deps = [
        ":smoke_test_base",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/models/ggnn:ggnn_node_classifier",
        "//labm8:app",
        "//third_party/py/numpy",
    ],
)
