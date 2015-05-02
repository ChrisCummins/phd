// -*- c-basic-offset: 8; -*-
#ifndef RANDOM_H_
#define RANDOM_H_

#include <cstdint>
#include <cstddef>

#include "./math.h"

// A random number generator for sampling a uniform distribution
// within a specific range.
class UniformDistribution {
 public:
    UniformDistribution(const Scalar min, const Scalar max,
                        const uint64_t seed = 7564231ULL)
            : divisor(scalarMax / (max - min)), min(min), seed(seed) {}

    // Return a new random number.
    Scalar inline operator()() {
            seed *= mult;

            // Generate a new random value in the range [0,max - min].
            const double r = seed % longMax / divisor;
            // Apply "-min" offset to value.
            return r + min;
    }

    const Scalar divisor;
    const Scalar min;
    uint64_t seed;

 private:
    // Constant values for random number generators.
    static const uint64_t rngMax;
    static const uint64_t longMax;
    static const Scalar scalarMax;
    static const uint64_t mult;
};

#endif  // RANDOM_H_
