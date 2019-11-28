#include "./tests.hpp"

#include <lm/meta-math>

TEST(MetaMath, basics) {
  static_assert(lm::metamath::add<10, 5>::value == 15, "error");
  static_assert(lm::metamath::add<-1, 5>::value == 4, "error");
}
