# Deep Reinforcement Learning Course.
# See: https://simoninithomas.github.io/Deep_reinforcement_learning_Course/

py_binary(
    name = "q_learning_frozen_lake",
    srcs = ["q_learning_frozen_lake.py"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//third_party/py/gym",
    ],
)

py_test(
    name = "q_learning_frozen_lake_test",
    srcs = ["q_learning_frozen_lake_test.py"],
    deps = [
        ":q_learning_frozen_lake",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)
