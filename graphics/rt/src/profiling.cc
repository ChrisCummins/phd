/*
 * Copyright (C) 2015, 2016 Chris Cummins.
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
#include "rt/profiling.h"

namespace rt {

namespace profiling {

namespace counters {

static std::atomic<Counter> objectsCount;
static std::atomic<Counter> lightsCount;
static std::atomic<Counter> traceCounter;
static std::atomic<Counter> rayCounter;

void incObjectsCount(const size_t n) { objectsCount += n; }

Counter getObjectsCount() { return static_cast<Counter>(objectsCount); }

void incLightsCount(const size_t n) { lightsCount += n; }

Counter getLightsCount() { return static_cast<Counter>(lightsCount); }

void incTraceCount(const size_t n) { traceCounter += n; }

Counter getTraceCount() { return static_cast<Counter>(traceCounter); }

void incRayCount(const size_t n) { rayCounter += n; }

Counter getRayCount() { return static_cast<Counter>(rayCounter); }

}  // namespace counters

}  // namespace profiling

}  // namespace rt
