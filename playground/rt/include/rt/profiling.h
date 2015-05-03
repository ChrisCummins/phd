// -*- c-basic-offset: 8; -*-
#ifndef RT_PROFILING_H_
#define RT_PROFILING_H_

#include <atomic>
#include <chrono>

#include "./math.h"

namespace rt {

namespace profiling {

// A profiling timer.
class Timer {
 public:
        // Create and start timer.
        inline Timer() : start(std::chrono::high_resolution_clock::now()) {}

        // Return the number of milliseconds.
        Scalar inline elapsed() {
                const std::chrono::high_resolution_clock::time_point end =
                                std::chrono::high_resolution_clock::now();
                return static_cast<Scalar>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        end - start).count() / 1e6);
        }

 private:
        const std::chrono::high_resolution_clock::time_point start;
};

// Counter data type.
typedef uint64_t Counter;

namespace counters {

// Counter for the number of objects.
void incObjectsCount();
Counter getObjectsCount();

// Counter for the number of lights.
void incLightsCount();
Counter getLightsCount();

// Counter for the number of traces.
void incTraceCount();
Counter getTraceCount();

// Counter for the number of rays.
void incRayCount();
Counter getRayCount();

}  // namespace counters

}  // namespace profiling

}  // namespace rt

#endif  // RT_PROFILING_H_
