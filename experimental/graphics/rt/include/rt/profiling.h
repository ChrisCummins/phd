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
#ifndef RT_PROFILING_H_
#define RT_PROFILING_H_

#include <atomic>
#include <chrono>

#include "rt/math.h"

namespace rt {

namespace profiling {

// A profiling timer.
class Timer {
 public:
  // Create and start timer.
  inline Timer() : start(std::chrono::high_resolution_clock::now()) {}

  // Return the number of milliseconds.
  auto inline elapsed() {
    const std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    return static_cast<Scalar>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1e6);
  }

 private:
  const std::chrono::high_resolution_clock::time_point start;
};

// Counter data type.
using Counter = uint64_t;

namespace counters {

// Counter for the number of objects.
void incObjectsCount(const size_t n = 1);
Counter getObjectsCount();

// Counter for the number of lights.
void incLightsCount(const size_t n = 1);
Counter getLightsCount();

// Counter for the number of traces.
void incTraceCount(const size_t n = 1);
Counter getTraceCount();

// Counter for the number of rays.
void incRayCount(const size_t n = 1);
Counter getRayCount();

}  // namespace counters

}  // namespace profiling

}  // namespace rt

#endif  // RT_PROFILING_H_
