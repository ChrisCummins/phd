# Doom-based AI Research Platform for Reinforcement Learning.
# See: https://github.com/mwydmuch/ViZDoom

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

py_library(
    name = "vizdoom",
    srcs = ["vizdoom.py"],
    deps = [
        # Note that the PyPi vizdoom package has a number of dependencies which
        # must be installed to build the package, see the official docs:
        # https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md
        requirement("vizdoom"),
    ],
)
