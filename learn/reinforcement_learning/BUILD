# Notes and experiments in learning to learn.

py_binary(
    name = "monte_carlo_control_blackjack",
    srcs = ["monte_carlo_control_blackjack.py"],
    deps = [
        "//labm8:app",
        "//third_party/py/gym",
    ],
)

py_binary(
    name = "notebooks",
    srcs = ["//notebooks"],
    deps = [
        ":monte_carlo_control_blackjack",
        "//labm8:app",
        "//third_party/py/jupyter",
    ],
)
