// -*- c-basic-offset: 8; -*-

#include "random.h"

// A random number generator for sampling a uniform distribution
// within a specific range.
class UniformDistribution {
  public:
    UniformDistribution(const Scalar min, const Scalar max,
                        const unsigned long long seed = 7564231ULL);

    Scalar operator()();

    const Scalar divisor;
    const Scalar min;
    unsigned long long seed;

  private:
    // Constant values for random number generators.
    static const unsigned long long rngMax;
    static const unsigned long long longMax;
    static const Scalar scalarMax;
    static const unsigned long long mult;
};
