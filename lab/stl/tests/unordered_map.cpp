#include "./tests.h"

#include <unordered_map>
#include <ustl/unordered_map>

#include <vector>
#include <ustl/vector>

#include <utility>

// constructors:

TEST(std_unordered_map, constructors) {
  // Empty:
  std::unordered_map<int, int> e1;
  std::unordered_map<int, int> e2(10);

  ASSERT_EQ(size_t(0), e1.size());
  ASSERT_TRUE(e1.empty());

  ASSERT_EQ(size_t(0), e2.size());
  ASSERT_TRUE(e2.empty());

  // Range:
  std::vector<std::pair<int, int>> in1{{1, 2}, {2, 3}, {3, 4}};
  std::unordered_map<int, int> ra1(in1.begin(), in1.end());

  ASSERT_EQ(size_t(3), ra1.size());
  for (auto& v : ra1)
    ASSERT_TRUE(v.first == v.second - 1);

  // Initialiazer list:
  std::unordered_map<int, int> il1({{1, 2}, {2, 3}, {3, 4}});

  ASSERT_EQ(size_t(3), il1.size());
  for (auto& v : il1)
    ASSERT_TRUE(v.first == v.second - 1);

  // Copy:
  auto cp1 = il1;

  ASSERT_EQ(size_t(3), cp1.size());
  for (auto& v : cp1)
    ASSERT_TRUE(v.first == v.second - 1);
}


TEST(ustl_unordered_map, constructors) {
  // Empty:
  ustl::unordered_map<int, int> e1;
  ASSERT_EQ(size_t(0), e1.size());
  ASSERT_TRUE(e1.empty());

  ustl::unordered_map<int, int> e2(10);
  ASSERT_EQ(size_t(0), e2.size());
  ASSERT_TRUE(e2.empty());

  // Range:
  ustl::vector<std::pair<int, int>> in1{{1, 2}, {2, 3}, {3, 4}};
  ustl::unordered_map<int, int> ra1(in1.begin(), in1.end());
  ASSERT_EQ(size_t(3), ra1.size());
  for (auto& v : ra1)
    ASSERT_TRUE(v.first == v.second - 1);

  // Initialiazer list:
  ustl::unordered_map<int, int> il1({{1, 2}, {2, 3}, {3, 4}});

  ASSERT_EQ(size_t(3), il1.size());
  for (auto& v : il1)
    ASSERT_TRUE(v.first == v.second - 1);

  // Copy:
  auto cp1 = il1;

  ASSERT_EQ(size_t(3), cp1.size());
  for (auto& v : cp1)
    ASSERT_TRUE(v.first == v.second - 1);
}


TEST(std_unordered_map, assignment) {
  std::unordered_map<int, int> src{{1, 2}, {3, 4}, {5, 6}};
  std::unordered_map<int, int> dst{{7, 8}};

  dst = src;
  ASSERT_TRUE(src == dst);
}

TEST(ustl_unordered_map, assignment) {
  ustl::unordered_map<int, int> src{{1, 2}, {3, 4}, {5, 6}};
  ustl::unordered_map<int, int> dst{{7, 8}};

  dst = src;
  ASSERT_TRUE(src == dst);
}
