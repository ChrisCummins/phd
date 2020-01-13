"""Debugging script to check installed tensorflow version and GPU status.

Usage: python ./overview.py 2>/dev/null
"""
import os

from third_party.py.tensorflow import tf


def main():
  print(
    """
TENSORFLOW
====================
"""
  )
  print("tf.__version__             ", tf.__version__)
  print("$CUDA_VISIBLE_DEVICES      ", os.environ.get("CUDA_VISIBLE_DEVICES"))
  print("tf.test.is_gpu_available() ", tf.test.is_gpu_available())


if __name__ == "__main__":
  main()
