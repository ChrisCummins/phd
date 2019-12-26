#include <unordered_map>
#include <ustl/unordered_map>
#include <ustl/vector>
#include <utility>
#include <vector>

#include "./tests.h"

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
  for (auto& v : ra1) ASSERT_TRUE(v.first == v.second - 1);

  // Initialiazer list:
  std::unordered_map<int, int> il1({{1, 2}, {2, 3}, {3, 4}});

  ASSERT_EQ(size_t(3), il1.size());
  for (auto& v : il1) ASSERT_TRUE(v.first == v.second - 1);

  // Copy:
  auto cp1 = il1;

  ASSERT_EQ(size_t(3), cp1.size());
  for (auto& v : cp1) ASSERT_TRUE(v.first == v.second - 1);
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
  for (auto& v : ra1) ASSERT_TRUE(v.first == v.second - 1);

  // Initialiazer list:
  ustl::unordered_map<int, int> il1({{1, 2}, {2, 3}, {3, 4}});

  ASSERT_EQ(size_t(3), il1.size());
  for (auto& v : il1) ASSERT_TRUE(v.first == v.second - 1);

  // Copy:
  auto cp1 = il1;

  ASSERT_EQ(size_t(3), cp1.size());
  for (auto& v : cp1) ASSERT_TRUE(v.first == v.second - 1);
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

///////////////
// Capacity: //
///////////////

TEST(std_unordered_map_capacity, empty) {
  std::unordered_map<int, int> l1;
  ASSERT_TRUE(l1.empty());
  l1[2] = 3;
  ASSERT_FALSE(l1.empty());
}

TEST(ustl_unordered_map_capacity, empty) {
  ustl::unordered_map<int, int> l1;
  ASSERT_TRUE(l1.empty());
  l1[2] = 3;
  ASSERT_FALSE(l1.empty());
}

TEST(std_unordered_map_capacity, size) {
  std::unordered_map<int, int> l1;
  ASSERT_EQ(0u, l1.size());
  l1[2] = 3;
  ASSERT_EQ(1u, l1.size());
  l1[2] = 4;
  ASSERT_EQ(1u, l1.size());
  l1[1] = 5;
  ASSERT_EQ(2u, l1.size());
}

TEST(std_unordered_map_capacity, max_size) {
  std::unordered_map<int, int> l1;
  ustl::unordered_map<int, int> l2 = {{1, 2}, {3, 4}, {5, 6}};

  ASSERT_GT(l1.max_size(), 1u);
  ASSERT_GT(l2.max_size(), 3u);
}

TEST(ustl_unordered_map_capacity, max_size) {
  ustl::unordered_map<int, int> l1;
  ustl::unordered_map<int, int> l2 = {{1, 2}, {3, 4}, {5, 6}};

  ASSERT_GT(l1.max_size(), 1u);
  ASSERT_GT(l2.max_size(), 3u);
}

////////////////
// Iterators: //
////////////////

TEST(std_unsorted_map_iterator, begin) {
  std::unordered_map<int, int> l1;
  ASSERT_TRUE(l1.begin() == l1.end());
  l1[1] = 2;
  ASSERT_EQ(1, (*l1.begin()).first);
  ASSERT_EQ(2, (*l1.begin()).second);
}

TEST(ustl_unsorted_map_iterator, begin) {
  ustl::unordered_map<int, int> l1;
  ASSERT_TRUE(l1.begin() == l1.end());
  l1[1] = 2;
  ASSERT_EQ(1, (*l1.begin()).first);
  ASSERT_EQ(2, (*l1.begin()).second);
}

/////////////////////
// Element Access: //
/////////////////////

TEST(std_unsorted_map_access, subscript) {
  std::unordered_map<char, int> l1;
  l1['a'] = 3;
  l1['a'] = 1;
  l1['b'] = 2;

  ASSERT_EQ(1, l1['a']);
  ASSERT_EQ(2, l1['b']);
}

TEST(ustl_unsorted_map_access, subscript) {
  ustl::unordered_map<char, int> l1;
  l1['a'] = 3;
  l1['a'] = 1;
  l1['b'] = 2;

  ASSERT_EQ(1, l1['a']);
  ASSERT_EQ(2, l1['b']);
}

TEST(std_unsorted_map_access, at) {
  std::unordered_map<char, int> l1;
  l1['a'] = 3;
  l1.at('a') = 1;
  l1['b'] = 2;

  ASSERT_EQ(1, l1.at('a'));
  ASSERT_EQ(2, l1.at('b'));
  try {
    l1.at('c') = 5;
    FAIL() << "std::out_of_range not thrown.";
  } catch (std::out_of_range e) {
  }
}

TEST(ustl_unsorted_map_access, at) {
  ustl::unordered_map<char, int> l1;
  l1['a'] = 3;
  l1.at('a') = 1;
  l1['b'] = 2;

  ASSERT_EQ(1, l1.at('a'));
  ASSERT_EQ(2, l1.at('b'));
  try {
    l1.at('c') = 5;
    FAIL() << "std::out_of_range not thrown.";
  } catch (std::out_of_range e) {
  }
}

/////////////////////
// Element Lookup: //
/////////////////////

TEST(std_unsorted_map_lookup, find) {
  std::unordered_map<int, char> l1{{1, 'a'}, {2, 'b'}, {3, 'c'}, {4, 'd'}};

  ASSERT_TRUE(l1.find(5) == l1.end());
  ASSERT_EQ('a', (*l1.find(1)).second);
  ASSERT_EQ('b', (*l1.find(2)).second);
}

TEST(ustl_unsorted_map_lookup, find) {
  ustl::unordered_map<int, char> l1{{1, 'a'}, {2, 'b'}, {3, 'c'}, {4, 'd'}};

  ASSERT_TRUE(l1.find(5) == l1.end());
  ASSERT_EQ('a', (*l1.find(1)).second);
  ASSERT_EQ('b', (*l1.find(2)).second);
}

TEST_MAIN();
