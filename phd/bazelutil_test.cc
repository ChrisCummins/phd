#include "phd/bazelutil.h"

#include "phd/test.h"

namespace phd {
namespace {

TEST(DataPathOrDie, DataPathExists) {
  BazelDataPathOrDie("phd/phd/test/data_file.txt");
}

}  // anonymous namespace
}  // namespace phd

TEST_MAIN();
