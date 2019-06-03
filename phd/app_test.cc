#include "phd/app.h"

#include "phd/test.h"

namespace phd {
namespace {

TEST(InitApp, DoesNotCrash) {
  int argc = 1;
  char* args[] = {"myapp"};
  char** argv = args;
  InitApp(&argc, &argv, "My usage string");
}

}  // anonymous namespace
}  // namespace phd

TEST_MAIN();
