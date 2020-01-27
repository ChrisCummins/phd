// This problem was asked by Google.
//
// The area of a circle is defined as πr^2. Estimate π to 3 decimal places using
// a Monte Carlo method.
//
// Hint: The basic equation of a circle is x^2 + y^2 = r^2.

#include "labm8/cpp/test.h"

float pi() {
  int c = 0;
  for (int xi = -1000; xi <= 1000; ++xi) {
    for (int yi = -1000; yi <= 1000; ++yi) {
      float x = xi / 1000., y = yi / 1000.;
      if (x * x + y * y < 1) {
        ++c;
      }
    }
  }

  return c / 1000000.;
}

TEST(EstimatePi, Pi) { ASSERT_NEAR(pi(), 3.141, 0.001); }

TEST_MAIN();
