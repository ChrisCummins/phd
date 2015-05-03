// -*- c-basic-offset: 8; -*-
#ifndef RT_PROFILING_H_
#define RT_PROFILING_H_

#include <atomic>

namespace rt {

namespace profiling {

// Profiling counter.
extern uint64_t objectsCount;

// A profiling counter that keeps track of how many times we've called
// Renderer::trace().
extern std::atomic<uint64_t> traceCounter;

// A profiling counter that keeps track of how many times we've
// contributed light to a ray.
extern std::atomic<uint64_t> rayCounter;

// Profiling counter.
extern uint64_t lightsCount;

}  // namespace profiling

}  // namespace rt

#endif  // RT_PROFILING_H_
