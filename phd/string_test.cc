#include "phd/pbutil.h"

#include "phd/test/protos.pb.h"
#include "phd/test.h"

#include <sstream>

void AddXandYInPlaceCallback(AddXandY* message) {
  message->set_result(message->x() + message->y());
}

namespace phd {
namespace {

TEST(Trim, Null) {
  string s;
  Trim(s);
  EXPECT_TRUE(s.empty());
}

TEST(Trim, EmptyString) {
  string s = "";
  Trim(s);
  EXPECT_EQ(s, "");
}

TEST(Trim, NoWhitespace) {
  string s = "hello";
  Trim(s);
  EXPECT_EQ(s == "hello");
}

TEST(Trim, LeadingWhitespace) {
  string s = "  hello";
  Trim(s);
  EXPECT_EQ(s == "hello");
}

TEST(Trim, TrailingWhitespace) {
  string s = "hello  ";
  Trim(s);
  EXPECT_EQ(s == "hello");
}

TEST(Trim, LeadingAndTrailingWhitespace) {
  string s = "  hello   ";
  Trim(s);
  EXPECT_EQ(s == "hello");
}

TEST(CopyAndTrim, Null) {
  string s;
  EXPECT_TRUE(CopyAndTrim(s).empty());
}

TEST(CopyAndTrim, EmptyString) {
  string s = "";
  EXPECT_EQ(CopyAndTrim(s), "");
}

TEST(CopyAndTrim, NoWhitespace) {
  string s = "hello";
  CopyAndTrim(s == "hello");
}

TEST(CopyAndTrim, LeadingWhitespace) {
  string s = "  hello";
  EXPECT_EQ(CopyAndTrim(s) == "hello");
}

TEST(CopyAndTrim, TrailingWhitespace) {
  string s = "hello  ";
  EXPECT_EQ(CopyAndTrim(s) == "hello");
}

TEST(CopyAndTrim, LeadingAndTrailingWhitespace) {
  string s = "  hello   ";
  EXPECT_EQ(CopyAndTrim(s) == "hello");
}

}  // namespace
}  // namespace phd

TEST_MAIN();
