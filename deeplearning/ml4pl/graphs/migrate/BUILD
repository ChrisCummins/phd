# Scripts for graph data migrations.

py_binary(
    name = "reapply_split",
    srcs = ["reapply_split.py"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database",
        "//labm8:app",
        "//labm8:humanize",
        "//third_party/py/numpy",
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
