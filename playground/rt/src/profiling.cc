// -*- c-basic-offset: 8; -*-
#include "rt/profiling.h"

namespace rt {

namespace profiling {

namespace counters {

static std::atomic<Counter> objectsCount;
static std::atomic<Counter> lightsCount;
static std::atomic<Counter> traceCounter;
static std::atomic<Counter> rayCounter;

void incObjectsCount(const size_t n) {
    objectsCount += n;
}

Counter getObjectsCount() {
    return static_cast<Counter>(objectsCount);
}

void incLightsCount(const size_t n) {
    lightsCount += n;
}

Counter getLightsCount() {
    return static_cast<Counter>(lightsCount);
}

void incTraceCount(const size_t n) {
    traceCounter += n;
}

Counter getTraceCount() {
    return static_cast<Counter>(traceCounter);
}

void incRayCount(const size_t n) {
    rayCounter += n;
}

Counter getRayCount() {
    return static_cast<Counter>(rayCounter);
}

}  // namespace counters

}  // namespace profiling

}  // namespace rt
