// -*- c-basic-offset: 8; -*-
#ifndef _RT_RANDOM_H_
#define _RT_RANDOM_H_

// A random number generator for sampling a uniform distribution
// within a specific range.
class UniformDistribution {
public:
    UniformDistribution(const Scalar min, const Scalar max,
                        const unsigned long long seed = 7564231ULL)
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
    unsigned long long seed;

private:
    // Constant values for random number generators.
    static const unsigned long long rngMax;
    static const unsigned long long longMax;
    static const Scalar scalarMax;
    static const unsigned long long mult;
};

// Set static member values:

const unsigned long long UniformDistribution::rngMax    = 4294967295ULL;
const unsigned long long UniformDistribution::longMax   = rngMax;
const Scalar             UniformDistribution::scalarMax = rngMax;
const unsigned long long UniformDistribution::mult      = 62089911ULL;

#endif  // _RT_RANDOM_H_
