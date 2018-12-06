# Deep Reinforcement Learning Course.
# See: https://simoninithomas.github.io/Deep_reinforcement_learning_Course/

py_binary(
    name = "deep_q_doom",
    srcs = ["deep_q_doom.py"],
    data = [
        "data/doom_config.cfg",
        "data/doom_scenario.wad",
    ],
    deps = [
        "//labm8:bazelutil",
        "//third_party/py/absl",
        "//third_party/py/numpy",
        "//third_party/py/scipy:scikit_image",
        "//third_party/py/tensorflow",
        "//third_party/py/vizdoom",
    ],
)

py_test(
    name = "deep_q_doom_test",
    srcs = ["deep_q_doom_test.py"],
    deps = [
        ":deep_q_doom",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

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
