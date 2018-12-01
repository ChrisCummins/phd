#include "phd/macros.h"
#include "phd/test.h"

TEST(Macros, LoggingMacros) {
  DEBUG("Number %d", 42);
  INFO("Number %d", 42);
  WARN("Number %d", 42);
  ERROR("Number %d", 42);

  // FATAL() causes the program to terminate.
  EXPECT_DEATH(FATAL("Number %d", 42), "Number 42");
}

TEST(Macros, Check) {
  CHECK(2 + 2 == 4);
  CHECK(true);

  EXPECT_DEATH(CHECK(2 + 2 == 5), "CHECK\\(2 \\+ 2 == 5\\) failed!");
  EXPECT_DEATH(CHECK(false), "CHECK\\(false\\) failed!");
}

TEST_MAIN();
