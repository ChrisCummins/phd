#include "./tests.h"

#include <algorithm>
#include <deque>

#include <stack>
#include <ustl/stack>

///////////////////
// Constructors: //
///////////////////

TEST(std_stack, constructors) {
  std::deque<int> q;

  std::stack<int> s1;
  std::stack<int> s2{q};
  std::stack<int> s3{s2};
}

TEST(ustl_stack, constructors) {
  std::deque<int> q;

  ustl::stack<int> s1;
  ustl::stack<int> s2{q};
  ustl::stack<int> s3{s2};
}

///////////////////////
// Member functions: //
///////////////////////

TEST(std_stack, basic_ops) {
  std::stack<int> s1;
  ASSERT_EQ(0u, s1.size());
  ASSERT_TRUE(s1.empty());
  s1.push(5);
  ASSERT_FALSE(s1.empty());
  ASSERT_EQ(1u, s1.size());
  ASSERT_EQ(5, s1.top());
  s1.pop();
  ASSERT_TRUE(s1.empty());
}

TEST(ustl_stack, basic_ops) {
  ustl::stack<int> s1;
  ASSERT_EQ(0u, s1.size());
  ASSERT_TRUE(s1.empty());
  s1.push(5);
  ASSERT_FALSE(s1.empty());
  ASSERT_EQ(1u, s1.size());
  ASSERT_EQ(5, s1.top());
  s1.pop();
  ASSERT_TRUE(s1.empty());
}

TEST(std_stack, emplace) {
  std::stack<int> s1;

  s1.emplace(5);
  ASSERT_EQ(5, s1.top());
}

TEST(ustl_stack, emplace) {
  ustl::stack<int> s1;

  s1.emplace(5);
  ASSERT_EQ(5, s1.top());
}

TEST(std_stack, swap) {
  std::stack<int> s1;
  std::stack<int> s2;

  s1.push(2);
  s1.push(1);
  s1.push(0);

  s2.push(3);

  s1.swap(s2);

  ASSERT_EQ(1u, s1.size());
  ASSERT_EQ(3u, s2.size());
  ASSERT_EQ(3, s1.top());
  ASSERT_EQ(0, s2.top());

  std::swap(s1, s2);

  ASSERT_EQ(3u, s1.size());
  ASSERT_EQ(1u, s2.size());
  ASSERT_EQ(0, s1.top());
  ASSERT_EQ(3, s2.top());
}

TEST(ustl_stack, swap) {
  ustl::stack<int> s1;
  ustl::stack<int> s2;

  s1.push(2);
  s1.push(1);
  s1.push(0);

  s2.push(3);

  s1.swap(s2);

  ASSERT_EQ(1u, s1.size());
  ASSERT_EQ(3u, s2.size());
  ASSERT_EQ(3, s1.top());
  ASSERT_EQ(0, s2.top());

  ustl::swap(s1, s2);  // NOLINT(build/include_what_you_use)

  ASSERT_EQ(3u, s1.size());
  ASSERT_EQ(1u, s2.size());
  ASSERT_EQ(0, s1.top());
  ASSERT_EQ(3, s2.top());
}

////////////////////////////////////
// Non-member function overloads: //
////////////////////////////////////

TEST(std_stack, relational_ops) {
  std::stack<int> s1;
  std::stack<int> s2;
  std::stack<int> s3;

  s1.push(3);
  s1.push(2);
  s1.push(1);

  s2.push(3);
  s2.push(2);
  s2.push(1);

  s3.push(-1);

  ASSERT_TRUE(s1 == s2);
  ASSERT_TRUE(s1 != s3);
  ASSERT_TRUE(s3 < s1);
  ASSERT_TRUE(s3 <= s1);
  ASSERT_TRUE(s1 <= s2);
  ASSERT_FALSE(s1 > s2);
  ASSERT_TRUE(s1 >= s2);
}

TEST(ustl_stack, relational_ops) {
  ustl::stack<int> s1;
  ustl::stack<int> s2;
  ustl::stack<int> s3;

  s1.push(3);
  s1.push(2);
  s1.push(1);

  s2.push(3);
  s2.push(2);
  s2.push(1);

  s3.push(-1);

  ASSERT_TRUE(s1 == s2);
  ASSERT_TRUE(s1 != s3);
  ASSERT_TRUE(s3 < s1);
  ASSERT_TRUE(s3 <= s1);
  ASSERT_TRUE(s1 <= s2);
  ASSERT_FALSE(s1 > s2);
  ASSERT_TRUE(s1 >= s2);
}
