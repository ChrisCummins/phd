from unittest import main

from labm8.tests.testutil import TestCase
from omnitune import opencl


class TestOpenCL(TestCase):

  # get_devices()
  def test_get_devices(self):
    self._test(True, isinstance(opencl.get_devices(), list))


if __name__ == '__main__':
  main()
