#include "learn/challenges/049_miu/miu.h"
#include "labm8/cpp/string.h"
#include "labm8/cpp/test.h"

namespace miu {
namespace {

TEST(Miu, SolveMiuMiu) {
  string input = "MI", output = "MI";
  std::vector<string> expected{"MI"};
  EXPECT_EQ(Solve(input, output), expected);
}

TEST(Miu, SolveMiMi) {
  string input = "MIU", output = "MIU";
  std::vector<string> expected{"MIU"};
  EXPECT_EQ(Solve(input, output), expected);
}

TEST(Miu, SolveMiMiu) {
  string input = "MI", output = "MIU";
  std::vector<string> expected{"MI", "MIU"};
  EXPECT_EQ(Solve(input, output), expected);
}

TEST(Miu, SolveMiMu) {
  string input = "MI", output = "MU";
  std::vector<string> expected;
  EXPECT_EQ(Solve(input, output, 100), expected);
}

TEST(Miu, SolveMiMiiii) {
  string input = "MI", output = "MIIII";
  std::vector<string> expected{"MI", "MII", "MIIII"};
  EXPECT_EQ(Solve(input, output, 100), expected);
}

}  // namespace
}  // namespace miu

TEST_MAIN();
