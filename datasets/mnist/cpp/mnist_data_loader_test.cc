#include "datasets/mnist/cpp/mnist_data_loader.h"
#include "labm8/cpp/test.h"

namespace mnist {
namespace {

TEST(MnistDataset, ReadMnistData) {
  MnistDataLoader loader;

  auto mnistOr = loader.Load();

  ASSERT_TRUE(mnistOr.ok());

  Mnist data = mnistOr.ValueOrDie();
  EXPECT_EQ(data.train.images.size(), 60000);
  EXPECT_EQ(data.train.labels.size(), 60000);

  EXPECT_EQ(data.test.images.size(), 10000);
  EXPECT_EQ(data.test.labels.size(), 10000);
}

}  // anonymous namespace
}  // namespace mnist

TEST_MAIN();
