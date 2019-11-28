"""Profile string concatenation vs join.

Results from a 2015 Intel desktop:

------------------------------------------------------------------------------------------ benchmark: 10 tests ------------------------------------------------------------------------------------------
Name (time in us)                 Min                   Max                Mean              StdDev              Median               IQR            Outliers  OPS (Kops/s)            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_string_concat[50]         4.6600 (1.0)         54.1710 (1.0)        4.7923 (1.0)        0.3439 (1.0)        4.7800 (1.0)      0.0430 (1.16)     252;2552      208.6688 (1.0)      103746           1
test_string_join[50]           6.2480 (1.34)        71.7430 (1.32)       6.5799 (1.37)       1.7051 (4.96)       6.3500 (1.33)     0.0420 (1.14)    2429;6060      151.9790 (0.73)     109987           1
test_string_concat[100]        8.2940 (1.78)       120.5870 (2.23)       8.4143 (1.76)       0.8073 (2.35)       8.3840 (1.75)     0.0370 (1.0)      362;4582      118.8454 (0.57)      98107           1
test_string_join[100]         11.6130 (2.49)       242.6640 (4.48)      12.0384 (2.51)       2.0309 (5.91)      11.8040 (2.47)     0.1260 (3.41)    1785;2951       83.0672 (0.40)      65262           1
test_string_concat[500]       44.6020 (9.57)       570.9100 (10.54)     45.6214 (9.52)       6.3595 (18.49)     44.8100 (9.37)     0.1290 (3.49)     419;1697       21.9195 (0.11)      18583           1
test_string_join[500]         54.8710 (11.77)      811.9590 (14.99)     56.4273 (11.77)      9.6693 (28.12)     55.4240 (11.59)    0.2770 (7.49)     310;1337       17.7219 (0.08)      15461           1
test_string_concat[1000]     109.4000 (23.48)      339.7100 (6.27)     111.1188 (23.19)      6.2108 (18.06)    110.2400 (23.06)    0.5015 (13.55)     239;558        8.9994 (0.04)       8507           1
test_string_join[1000]       110.9470 (23.81)      197.0040 (3.64)     112.2423 (23.42)      3.0104 (8.75)     111.6790 (23.36)    0.3720 (10.05)     381;862        8.9093 (0.04)       8532           1
test_string_join[5000]       557.6000 (119.66)   4,877.7000 (90.04)    563.9064 (117.67)   104.1281 (302.80)   560.0220 (117.16)   1.5847 (42.83)       3;195        1.7733 (0.01)       1725           1
test_string_concat[5000]     610.3570 (130.98)     773.2060 (14.27)    622.5233 (129.90)     8.7775 (25.52)    621.3145 (129.98)   4.1330 (111.71)    183;192        1.6064 (0.01)       1622           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

There is essentially no difference in performance. Maybe concat is slightly
faster (!).
"""
import pytest

from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

MODULE_UNDER_TEST = None  # No coverage.


@pytest.mark.parametrize("iterations", [50, 100, 500, 1000, 5000])
def test_string_concat(benchmark, iterations):
  """Benchmark string concatenation."""

  def Benchmark():
    """Benchmark inner loop."""
    a = "a"
    for _ in range(iterations):
      a += "a"
    return a

  benchmark(Benchmark)


@pytest.mark.parametrize("iterations", [50, 100, 500, 1000, 5000])
def test_string_join(benchmark, iterations):
  """Benchmark string concatenation."""

  def Benchmark():
    """Benchmark inner loop."""
    a = ["a"]
    for _ in range(iterations):
      a.append("a")
    return "".join(a)

  benchmark(Benchmark)


if __name__ == "__main__":
  test.Main()
