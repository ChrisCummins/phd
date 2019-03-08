# Deep Reinforcement Learning Course.
# See: https://simoninithomas.github.io/Deep_reinforcement_learning_Course/

py_binary(
    name = "q_learning_frozen_lake",
    srcs = ["q_learning_frozen_lake.py"],
    deps = [
        "//third_party/py/absl",
        "//third_party/py/gym",
        "//third_party/py/humanize",
    ],
)

py_test(
    name = "q_learning_frozen_lake_test",
    srcs = ["q_learning_frozen_lake_test.py"],
    deps = [
        ":q_learning_frozen_lake",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
