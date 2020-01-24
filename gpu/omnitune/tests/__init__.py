import json
import unittest


# Extension of unittest's TestCase.
class TestCase(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(TestCase, self).__init__(*args, **kwargs)
    self.stencil_gaussian_kernel = open(
      "tests/data/stencil-gaussian-kernel.cl"
    ).read()
    self.stencil_gaussian_kernel_user = open(
      "tests/data/stencil-gaussian-kernel-user.cl"
    ).read()
    self.stencil_gaussian_kernel_bc = open(
      "tests/data/stencil-gaussian-kernel.bc"
    ).read()
    self.stencil_gaussian_kernel_ic = open(
      "tests/data/stencil-gaussian-kernel.ic"
    ).read()
    self.stencil_gaussian_kernel_ic_json = json.loads(
      open("tests/data/stencil-gaussian-kernel.ic.json").read()
    )
    self.stencil_gaussian_kernel_ratios_json = json.loads(
      open("tests/data/stencil-gaussian-kernel-ratios.json").read()
    )

  @staticmethod
  def _print(obj, **kwargs):
    if hasattr(obj, "__iter__"):
      if isinstance(obj, dict):
        print(obj, **kwargs)
      elif not isinstance(obj, str):
        print([str(x) for x in obj], **kwargs)
    else:
      print(str(obj), **kwargs)

  # A convenience method to assert that expected result equals
  # actual result. The benefit over just calling assertTrue() is
  # that it prints the expected and actual values if the test fails.
  def _test(self, expected, actual):
    print("Expected: ", end="")
    self._print(expected)
    print("Actual:   ", end="")
    self._print(actual)
    try:
      self.assertTrue(actual == expected)
      print("OK\n")
    except AssertionError as e:
      print("FAIL")
      raise (e)
