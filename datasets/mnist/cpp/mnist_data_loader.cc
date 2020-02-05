#include "datasets/mnist/cpp/mnist_data_loader.h"

#include "labm8/cpp/bazelutil.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status_macros.h"

#include <fstream>

namespace mnist {

namespace {

int ReverseInt(int i) {
  unsigned char c1, c2, c3, c4;

  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;

  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

labm8::StatusOr<std::vector<Image>> ReadImages(const string& path) {
  std::ifstream file(path);
  CHECK(file.is_open()) << "Failed to open file: " << path;

  int magicNumber = 0;
  file.read((char*)&magicNumber, 4);
  magicNumber = ReverseInt(magicNumber);
  CHECK(magicNumber == 0x803)
      << "Expected magic number for " << path
      << " (0x803) does not match the number found in the file (" << magicNumber
      << ")";

  int numImages;
  file.read((char*)&numImages, 4);
  numImages = ReverseInt(numImages);
  CHECK(numImages > 0) << "Invalid image count in " << path << " (" << numImages
                       << ")";
  CHECK(numImages <= 60000)
      << "Expected images in " << path << " (" << numImages << ")";

  std::vector<Image> images;
  images.reserve(numImages);

  int rowCount;
  file.read((char*)&rowCount, 4);
  rowCount = ReverseInt(rowCount);
  CHECK(rowCount == 28) << "Expected 28 rows in " << path
                        << ", found: " << rowCount;

  int colCount;
  file.read((char*)&colCount, 4);
  colCount = ReverseInt(colCount);
  CHECK(colCount == 28) << "Expected 28 columns in " << path
                        << ", found: " << colCount;

  for (int i = 0; i < numImages; ++i) {
    Image image;
    file.read((char*)image.data(), 28 * 28);
    images.push_back(image);
  }

  return images;
}

labm8::StatusOr<std::vector<uint8_t>> ReadLabels(const string& path) {
  std::ifstream file(path);
  CHECK(file.is_open()) << "Failed to open file: " << path;

  int magicNumber = 0;
  file.read((char*)&magicNumber, 4);
  magicNumber = ReverseInt(magicNumber);
  CHECK(magicNumber == 0x801)
      << "Expected magic number for " << path
      << " (0x801) does not match the number found in the file (" << magicNumber
      << ")";

  int numImages;
  file.read((char*)&numImages, 4);
  numImages = ReverseInt(numImages);
  CHECK(numImages > 0) << "Invalid image count in " << path << " (" << numImages
                       << ")";
  CHECK(numImages <= 60000)
      << "Expected images in " << path << " (" << numImages << ")";

  std::vector<uint8_t> labels;
  labels.reserve(numImages);

  for (int i = 0; i < numImages; ++i) {
    uint8_t label;
    file.read((char*)&label, 1);
    labels.push_back(label);
  }

  return labels;
}

labm8::StatusOr<Mnist> ReadMnistData() {
  boost::filesystem::path train_images_path;
  boost::filesystem::path train_labels_path;
  boost::filesystem::path test_images_path;
  boost::filesystem::path test_labels_path;

  ASSIGN_OR_RETURN(
      train_images_path,
      labm8::BazelDataPath("phd/datasets/mnist/mnist_train_images.data"));
  ASSIGN_OR_RETURN(
      test_images_path,
      labm8::BazelDataPath("phd/datasets/mnist/mnist_test_images.data"));
  ASSIGN_OR_RETURN(
      train_labels_path,
      labm8::BazelDataPath("phd/datasets/mnist/mnist_train_labels.data"));
  ASSIGN_OR_RETURN(
      test_labels_path,
      labm8::BazelDataPath("phd/datasets/mnist/mnist_test_labels.data"));

  std::vector<Image> train_images;
  std::vector<Image> test_images;
  std::vector<uint8_t> train_labels;
  std::vector<uint8_t> test_labels;

  ASSIGN_OR_RETURN(train_images, ReadImages(train_images_path.string()));
  ASSIGN_OR_RETURN(test_images, ReadImages(test_images_path.string()));
  ASSIGN_OR_RETURN(train_labels, ReadLabels(train_labels_path.string()));
  ASSIGN_OR_RETURN(test_labels, ReadLabels(test_labels_path.string()));

  CHECK(train_images.size() == train_labels.size());
  CHECK(test_images.size() == test_labels.size());

  LabeledImages train{train_images, train_labels};
  LabeledImages test{test_images, test_labels};

  return Mnist({train, test});
}

}  // anonymous namespace

labm8::StatusOr<Mnist> MnistDataLoader::Load() {
  if (!data_) {
    auto dataOr = ReadMnistData();
    if (dataOr.ok()) {
      Mnist* data = new Mnist();
      data_ = data;
      *data_ = dataOr.ValueOrDie();
    } else {
      status_ = dataOr.status();
    }
  }

  if (!status_.ok()) {
    return status_;
  }

  return *data_;
}

}  // namespace mnist
