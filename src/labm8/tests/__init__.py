import unittest

# Extension of unittest's TestCase.
class TestCase(unittest.TestCase):

    # A convenience method to assert that expected result equals
    # actual result. The benefit over just calling assertTrue() is
    # that it prints the expected and actual values if the test fails.
    def _test(self, expected, actual):
        print("Expected:", expected)
        print("Actual:", actual)
        self.assertTrue(actual == expected)
