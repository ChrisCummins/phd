// -*- c-basic-offset: 8; -*-
#ifndef RT_RANDOM_H_
#define RT_RANDOM_H_

#include <cstdint>
#include <cstddef>

#include "./math.h"

namespace rt {

// Data type for seeding random number generators.
typedef uint64_t Seed;

// A random number generator for sampling a uniform distribution
// within a specific range.
class UniformDistribution {
 public:
    inline UniformDistribution(const Scalar _min, const Scalar _max,
                               const Seed _seed = 7564231ULL)
            : divisor(scalarMax / (_max - _min)),
                      min(_min),  // NOLINT(build/include_what_you_use)
                      seed(_seed) {}

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

// A generator for sampling random points over a disk.
class UniformDiskDistribution {
 public:
        inline UniformDiskDistribution(const Scalar _radius,
                                       const Seed _seed = 7564231ULL)
                : angle(UniformDistribution(0, 2 * M_PI, _seed)),
                  rand01(UniformDistribution(0, 1, _seed)),
                  radius(_radius) {}

        // Return a random point on the disk, with the vector x and y
        // components corresponding to the x and y coordinates of the
        // point within the disk.
        Vector inline operator()() {
                const Scalar theta = angle();
                const Scalar distance = radius * sqrt(rand01());

                const Scalar x = distance * cos(theta);
                const Scalar y = distance * sin(theta);

                return Vector(x, y, 0);
        }

 private:
        UniformDistribution angle;
        UniformDistribution rand01;
        const Scalar radius;
};

}  // namespace rt

#endif  // RT_RANDOM_H_
