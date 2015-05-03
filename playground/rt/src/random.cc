// -*- c-basic-offset: 8; -*-

#include "rt/random.h"

namespace rt {

// Set static member values:

const uint64_t UniformDistribution::rngMax    = 4294967295ULL;
const uint64_t UniformDistribution::longMax   = rngMax;
const Scalar   UniformDistribution::scalarMax = rngMax;
const uint64_t UniformDistribution::mult      = 62089911ULL;

}  // namespace rt
