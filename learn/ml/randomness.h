#pragma once

#include <array>
#include <cmath>
#include <cstdlib>

namespace ml {

// Return a random value in the range [0,1].
template <typename T>
T Rand() {
  return static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
}

// Fill an array with random values in the range [0, sqrt(N)]. This is provides
// a variance over all N values of 1.
template <typename T, size_t N, size_t M>
void ArrayFillRandVarOne(std::array<T, N>* X) {
  T normalizer = sqrt(1.0 / M);

  for (size_t i = 0; i < N; ++i) {
    (*X)[i] = Rand<float>() / normalizer;
  }
}

}  // namespace ml
