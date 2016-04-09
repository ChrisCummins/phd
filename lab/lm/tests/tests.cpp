#include "./tests.hpp"

bool inverse_comp(const int &a, const int &b) { return a > b; }

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
