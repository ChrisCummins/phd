// -*- c-basic-offset: 8; -*-
#include "rt/profiling.h"

namespace rt {

namespace profiling {

// Profiling counter.
uint64_t objectsCount;

// Profiling counter.
uint64_t lightsCount;

// A profiling counter that keeps track of how many times we've called
// Renderer::trace().
std::atomic<uint64_t> traceCounter;

// A profiling counter that keeps track of how many times we've
// contributed light to a ray.
std::atomic<uint64_t> rayCounter;

}  // namespace profiling

}  // namespace rt
