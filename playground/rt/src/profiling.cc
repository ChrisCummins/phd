// -*- c-basic-offset: 8; -*-
#include "rt/profiling.h"

namespace rt {

namespace profiling {

namespace counters {

static std::atomic<Counter> objectsCount;
static std::atomic<Counter> lightsCount;
static std::atomic<Counter> traceCounter;
static std::atomic<Counter> rayCounter;

void incObjectsCount() {
    objectsCount++;
}

Counter getObjectsCount() {
    return static_cast<Counter>(objectsCount);
}

void incLightsCount() {
    lightsCount++;
}

Counter getLightsCount() {
    return static_cast<Counter>(lightsCount);
}

void incTraceCount() {
    traceCounter++;
}

Counter getTraceCount() {
    return static_cast<Counter>(traceCounter);
}

void incRayCount() {
    rayCounter++;
}

Counter getRayCount() {
    return static_cast<Counter>(rayCounter);
}

}  // namespace counters

}  // namespace profiling

}  // namespace rt
