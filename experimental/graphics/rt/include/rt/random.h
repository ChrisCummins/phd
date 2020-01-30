/* -*-c++-*-
 *
 * Copyright (C) 2015-2020 Chris Cummins.
 *
 * This file is part of rt.
 *
 * rt is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at
 * your option) any later version.
 *
 * rt is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
 * License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with rt.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef RT_RANDOM_H_
#define RT_RANDOM_H_

#include <cstdint>
#include <limits>

#include "rt/math.h"

namespace rt {

// Random number seed:
using seed = uint64_t;

// A random number generator for sampling a uniform distribution
// within a specific range.
template <typename T>
class UniformDistribution {
 public:
  UniformDistribution(const T& min, const T& max,
                      const seed seedval = seed{7564231})
      : _divisor(std::numeric_limits<T>::max() / (max - min)),
        _min(min),
        _seed(seedval) {}  // NOLINT(build/include_what_you_use)

  // Generate a new random value in the range [0,max - min].
  auto operator()() {
    _seed *= 62089911ULL;  // magic value
    return (_seed % std::numeric_limits<seed>::max() / _divisor) + _min;
  }

  const T _divisor;
  const T _min;
  seed _seed;
};

// A generator for sampling random points over a disk.
template <typename T>
class UniformDiskDistribution {
 public:
  UniformDiskDistribution(const T radius, const seed seed1 = seed{7564231},
                          const seed seed2 = seed{7564231})
      : angle(0, 2 * M_PI, seed1), rand01(0, 1, seed2), _radius(radius) {}

  // Return a random point on the disk, with the vector x and y
  // components corresponding to the x and y coordinates of the
  // point within the disk.
  auto operator()() {
    const T theta = angle();
    const T distance = _radius * sqrt(rand01());

    const T x = distance * cos(theta);
    const T y = distance * sin(theta);

    return Vector(x, y);
  }

 private:
  UniformDistribution<T> angle;
  UniformDistribution<T> rand01;
  const T _radius;
};

}  // namespace rt

#endif  // RT_RANDOM_H_
