#include "phd/string.h"

#include "phd/test.h"


namespace phd {
namespace {

TEST(TrimTest, Null) {
  string s;
  Trim(s);
  EXPECT_TRUE(s.empty());
}

TEST(TrimTest, EmptyString) {
  string s = "";
  Trim(s);
  EXPECT_EQ(s, "");
}

TEST(TrimTest, NoWhitespace) {
  string s = "hello";
  Trim(s);
  EXPECT_EQ(s, "hello");
}

TEST(TrimTest, LeadingWhitespace) {
  string s = "  hello";
  Trim(s);
  EXPECT_EQ(s, "hello");
}

TEST(TrimTest, TrailingWhitespace) {
  string s = "hello  ";
  Trim(s);
  EXPECT_EQ(s, "hello");
}

TEST(TrimTest, LeadingAndTrailingWhitespace) {
  string s = "  hello   ";
  Trim(s);
  EXPECT_EQ(s, "hello");
}

TEST(CopyAndTrimTest, Null) {
  string s;
  EXPECT_TRUE(CopyAndTrim(s).empty());
}

TEST(CopyAndTrimTest, EmptyString) {
  string s = "";
  EXPECT_EQ(CopyAndTrim(s), "");
}

TEST(CopyAndTrimTest, NoWhitespace) {
  string s = "hello";
  EXPECT_EQ(CopyAndTrim(s), "hello");
}

TEST(CopyAndTrimTest, LeadingWhitespace) {
  string s = "  hello";
  EXPECT_EQ(CopyAndTrim(s), "hello");
}

TEST(CopyAndTrimTest, TrailingWhitespace) {
  string s = "hello  ";
  EXPECT_EQ(CopyAndTrim(s), "hello");
}

TEST(CopyAndTrimTest, LeadingAndTrailingWhitespace) {
  string s = "  hello   ";
  EXPECT_EQ(CopyAndTrim(s), "hello");
}

TEST(ToCamelCase, EmptyString) {
  EXPECT_EQ(ToCamelCase(""), "");
}

TEST(ToCamelCase, Hello) {
  EXPECT_EQ(ToCamelCase("hello"), "Hello");
  EXPECT_EQ(ToCamelCase("Hello"), "Hello");
  EXPECT_EQ(ToCamelCase("HELLO"), "Hello");
}

TEST(ToCamelCase, LeadingWhitespace) {
  EXPECT_EQ(ToCamelCase("   hello"), "Hello");
}

TEST(ToCamelCase, TrailingWhitespace) {
  EXPECT_EQ(ToCamelCase("hello   "), "Hello");
}

TEST(ToCamelCase, MultipleComponents) {
  EXPECT_EQ(ToCamelCase("hello world"), "HelloWorld");
}

TEST(ToCamelCase, MultipleInnerWhitespace) {
  EXPECT_EQ(ToCamelCase("hello    world"), "HelloWorld");
}

}  // namespace
}  // namespace phd

TEST_MAIN();
