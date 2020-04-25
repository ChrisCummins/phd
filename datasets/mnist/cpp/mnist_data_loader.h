#pragma once

#include <array>
#include <vector>
#include "labm8/cpp/statusor.h"
#include "labm8/cpp/string.h"

namespace mnist {

using Image = std::array<uint8_t, 28 * 28>;

// A set of <image, label> pairs.
struct LabeledImages {
  std::vector<Image> images;
  std::vector<uint8_t> labels;
};

// The MNIST dataset, divided into training and test data.
struct Mnist {
  LabeledImages train;
  LabeledImages test;
};

// A data loader for the MNIST dataset.
//
// Usage:
//    auto mnist = MnistDataLoader().Load().ValueOrDie();
class MnistDataLoader {
 public:
  MnistDataLoader() : data_(nullptr), status_(labm8::Status::OK) {}

  ~MnistDataLoader() {
    if (data_) {
      delete data_;
    }
  }

  // Load the MNIST dataset.
  labm8::StatusOr<Mnist> Load();

 private:
  Mnist* data_;
  labm8::Status status_;
};

}  // namespace mnist
