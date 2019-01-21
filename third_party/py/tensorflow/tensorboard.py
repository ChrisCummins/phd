"""Run Tensorboard main method. This script never terminates.

Usage:

    bazel run //third_party/py/tensorflow:tensorboard

Use this as a convenience binary to ensure that the tensorboard being invoked
is the same one as used by bazel.
"""
from tensorboard import main as tensorboard_main


if __name__ == '__main__':
  tensorboard_main.run_main()
