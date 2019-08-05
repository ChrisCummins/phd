#include "labm8/cpp/bazelutil.h"

#include "labm8/cpp/test.h"

namespace labm8 {
namespace {

TEST(DataPathOrDie, DataPathExists) {
  BazelDataPathOrDie("phd/phd/test/data_file.txt");
}

}  // anonymous namespace
}  // namespace labm8

TEST_MAIN();
