#include "./tests.hpp"

#include <lm/meta-math>

TEST(MetaMath, basics) {
    static_assert(lm::metamath::add<10, 5>::value == 15);
    static_assert(lm::metamath::add<-1, 5>::value == 4);
}
