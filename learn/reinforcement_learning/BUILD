# Notes and experiments in learning to learn.

py_binary(
    name = "monte_carlo_control_blackjack",
    srcs = ["monte_carlo_control_blackjack.py"],
    deps = [
        "//third_party/py/absl",
        "//third_party/py/gym",
    ],
)

py_binary(
    name = "notebooks",
    srcs = ["//labm8:notebooks"],
    deps = [
        ":monte_carlo_control_blackjack",
        "//third_party/py/absl",
        "//third_party/py/jupyter",
    ],
)
