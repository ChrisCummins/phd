#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <algorithm>
#include <ustl/algorithm>

#include <array>
#include <ustl/array>

#include <vector>
#include <ustl/vector>

//
// Helper functions & objects.
//

template<typename T>
class InverseComp {
 public:
  bool operator()(const T &a, const T &b) { return a > b; }
};

class Comparable {
 public:
  Comparable() : data(0) {}
  explicit Comparable(const int _data) : Comparable(_data, 0) {}
  Comparable(const int _data, const int _nc) : data(_data), nc(_nc) {}
  ~Comparable() {}

  int data;
  int nc;
  bool operator<(const Comparable &rhs) const { return data < rhs.data; }
  bool operator==(const Comparable &rhs) const { return data == rhs.data; }
};

static bool inverse_comp(const int &a, const int &b) { return a > b; }

// Algorithm tests

// Merge

TEST(algorithm, merge) {
  ustl::array<int, 5> a{1, 3, 5, 7, 9};
  ustl::array<int, 5> b{2, 4, 6, 8, 10};
  ustl::array<int, 10> c;
  const ustl::array<int, 10> sorted{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  ustl::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin());

  for (size_t i = 0; i < c.size(); i++)
    ASSERT_EQ(c[i], sorted[i]);
}

TEST(algorithm, merge_comp_lambda) {
  ustl::array<int, 5> a{9, 7, 5, 3, 1};
  ustl::array<int, 5> b{10, 8, 6, 4, 2};
  ustl::array<int, 10> c;
  const ustl::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(),
              [](int x, int y) { return x > y; });

  for (size_t i = 0; i < c.size(); i++)
    ASSERT_EQ(c[i], sorted[i]);
}

TEST(algorithm, merge_comp_funcobj) {
  ustl::array<int, 5> a{9, 7, 5, 3, 1};
  ustl::array<int, 5> b{10, 8, 6, 4, 2};
  ustl::array<int, 10> c;
  const ustl::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(),
              InverseComp<int>());

  for (size_t i = 0; i < c.size(); i++)
    ASSERT_EQ(c[i], sorted[i]);
}

TEST(algorithm, merge_comp_funcptr) {
  ustl::array<int, 5> a{9, 7, 5, 3, 1};
  ustl::array<int, 5> b{10, 8, 6, 4, 2};
  ustl::array<int, 10> c;
  const ustl::array<int, 10> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::merge(a.begin(), a.end(), b.begin(), b.end(), c.begin(),
              inverse_comp);

  for (size_t i = 0; i < c.size(); i++)
    ASSERT_EQ(c[i], sorted[i]);
}

// Sort

TEST(algorithm, sort) {
  ustl::vector<int> a{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  ustl::vector<int> b{9, 8, 7, 6, 5, 4, 3, 2, 1};
  const ustl::vector<int> sorted{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  ustl::sort(a.begin(), a.end());
  ustl::sort(b.begin(), b.end());

  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], sorted[i]);

  for (size_t i = 0; i < b.size(); i++)
    ASSERT_EQ(b[i], sorted[i]);
}

TEST(algorithm, sort_comp_lambda) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const ustl::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::sort(a.begin(), a.end(), [](int x, int y) { return x > y; });

  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], sorted[i]);
}

TEST(algorithm, sort_comp_funcobj) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const ustl::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::sort(a.begin(), a.end(), InverseComp<int>());

  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], sorted[i]);
}

TEST(algorithm, sort_comp_funcptr) {
  ustl::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const ustl::vector<int> sorted{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  ustl::sort(a.begin(), a.end(), &inverse_comp);

  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], sorted[i]);
}

// Stable sort

TEST(algorithm, stable_sort) {
  ustl::vector<Comparable> a{
    Comparable(0, 0),
    Comparable(2),
    Comparable(0, 1),
    Comparable(1)
  };

  ustl::stable_sort(a.begin(), a.end());

  ASSERT_EQ(a[0].data, 0);
  ASSERT_EQ(a[0].nc,   0);
  ASSERT_EQ(a[1].data, 0);
  ASSERT_EQ(a[1].nc,   1);
  ASSERT_EQ(a[2].data, 1);
  ASSERT_EQ(a[3].data, 2);
}

// is_sorted()

TEST(algorithm, is_sorted) {
  ustl::vector<int> a{1, 2, 3, 4, 5};
  ustl::vector<int> b{5, 4, 3, 1, 2};
  ustl::vector<int> c{1, 2, 3, 5, 4};
  ustl::vector<int> d{1, 1, 1, 1, 1};

  ASSERT_TRUE(ustl::is_sorted(a.begin(), a.end()));
  ASSERT_FALSE(ustl::is_sorted(b.begin(), b.end()));
  ASSERT_FALSE(ustl::is_sorted(c.begin(), c.end()));
  ASSERT_TRUE(ustl::is_sorted(d.begin(), d.end()));
}

// Array tests

TEST(array, size) {
  ustl::array<int, 10> a;
  ustl::array<char, 1> b;
  ustl::array<int, 100> c;
  ustl::array<int, 0> d;

  ASSERT_EQ(10u, a.size());
  ASSERT_EQ(1u, b.size());
  ASSERT_EQ(100u, c.size());
  ASSERT_EQ(0u, d.size());
}

TEST(array, max_size) {
  ustl::array<int, 10> a;
  ustl::array<char, 1> b;
  ustl::array<int, 100> c;
  ustl::array<int, 0> d;

  ASSERT_EQ(10u, a.max_size());
  ASSERT_EQ(a.size(), a.max_size());
  ASSERT_EQ(1u, b.max_size());
  ASSERT_EQ(b.size(), b.max_size());
  ASSERT_EQ(100u, c.max_size());
  ASSERT_EQ(c.size(), c.max_size());
  ASSERT_EQ(0u, d.max_size());
  ASSERT_EQ(d.size(), d.max_size());
}

TEST(array, empty) {
  ustl::array<int, 10> a;
  ustl::array<char, 1> b;
  ustl::array<int, 100> c;
  ustl::array<int, 0> d;

  ASSERT_EQ(false, a.empty());
  ASSERT_EQ(false, b.empty());
  ASSERT_EQ(false, c.empty());
  ASSERT_EQ(true,  d.empty());
}

TEST(array, subscript) {
  ustl::array<int, 2> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;

  // Test values:
  ASSERT_EQ(1, a[0]);
  ASSERT_EQ(2, a[1]);
}

TEST(array, at) {
  ustl::array<int, 2> a;
  ustl::array<int, 0> b;

  // Set values:
  a[0] = 1;
  a[1] = 2;

  // Test values:
  ASSERT_EQ(1, a.at(0));
  ASSERT_EQ(2, a.at(1));

  // Out of range tests:
  try {
    a.at(2) = 3;
    FAIL();
  } catch (std::out_of_range) {}

  try {
    b.at(0) = 1;
    FAIL();
  } catch (std::out_of_range) {}
}

TEST(array, front) {
  ustl::array<int, 3> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;

  // Test values:
  ASSERT_EQ(1, a.front());
}

TEST(array, back) {
  ustl::array<int, 3> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;

  // Test values:
  ASSERT_EQ(3, a.back());
}

TEST(array, data) {
  ustl::array<int, 3> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;

  // Get pointer:
  auto *d = a.data();

  // Test values:
  ASSERT_EQ(1, d[0]);
  ASSERT_EQ(2, d[1]);
  ASSERT_EQ(3, d[2]);
}

// Vector tests

TEST(vector, size) {
  ustl::vector<int> a;
  ustl::vector<int> b(5);
  ustl::vector<double> c(6, 3.5f);
  ustl::vector<char> d = { 'a', 'b', 'c' };

  ASSERT_EQ(0u, a.size());
  ASSERT_EQ(5u, b.size());
  ASSERT_EQ(6u, c.size());
  ASSERT_EQ(3u, d.size());
}

TEST(vector, capacity) {
  ustl::vector<int> a;
  ustl::vector<int> b(5);
  ustl::vector<double> c(6, 3.5f);
  ustl::vector<char> d = { 'a', 'b', 'c' };

  ASSERT_TRUE(a.capacity() >= a.size());
  ASSERT_TRUE(b.capacity() >= b.size());
  ASSERT_TRUE(c.capacity() >= c.size());
  ASSERT_TRUE(d.capacity() >= d.size());
}

TEST(vector, max_size) {
  ustl::vector<int> a;
  ustl::vector<int> b(5);
  ustl::vector<double> c(6, 3.5f);
  ustl::vector<char> d = { 'a', 'b', 'c' };

  ASSERT_TRUE(a.size() < a.max_size());
  ASSERT_TRUE(b.size() < b.max_size());
  ASSERT_TRUE(c.size() < c.max_size());

  ASSERT_EQ(a.max_size(), b.max_size());
}

TEST(vector, constructorValues) {
  ustl::vector<int> a(3);
  ustl::vector<double> b(3, 3.5f);
  ustl::vector<char> c = { 'a', 'b', 'c' };

  for (size_t i = 0; i < 3; i++)
    ASSERT_EQ(0, a[i]);

  for (size_t i = 0; i < 3; i++)
    ASSERT_EQ(3.5f, b[i]);

  ASSERT_EQ('a', c[0]);
  ASSERT_EQ('b', c[1]);
  ASSERT_EQ('c', c[2]);
}

TEST(vector, front) {
  ustl::vector<int> a(3);
  ustl::vector<double> b(3, 3.5f);
  ustl::vector<char> c = { 'a', 'b', 'c' };

  ASSERT_EQ(0, a.front());
  ASSERT_EQ(3.5, b.front());
  ASSERT_EQ('a', c.front());
}

TEST(vector, back) {
  ustl::vector<int> a(3);
  ustl::vector<double> b(3, 3.5f);
  ustl::vector<char> c = { 'a', 'b', 'c' };

  ASSERT_EQ(0, a.back());
  ASSERT_EQ(3.5, b.back());
  ASSERT_EQ('c', c.back());
}

TEST(vector, at) {
  ustl::vector<int> a(3);
  ustl::vector<double> b(3, 3.5f);
  ustl::vector<char> c = { 'a', 'b', 'c' };

  for (size_t i = 0; i < 3; i++) {
    ASSERT_EQ(0, a.at(i));
    ASSERT_EQ(a[i], a.at(i));
  }
  try {
    a.at(3);
    FAIL();
  } catch (std::out_of_range &e) {}

  for (size_t i = 0; i < 3; i++) {
    ASSERT_EQ(3.5f, b.at(i));
    ASSERT_EQ(b[i], b.at(i));
  }
  try {
    b.at(3);
    FAIL();
  } catch (std::out_of_range &e) {}

  ASSERT_EQ('a', c.at(0));
  ASSERT_EQ('b', c.at(1));
  ASSERT_EQ('c', c.at(2));
  try {
    c.at(3);
    FAIL();
  } catch (std::out_of_range &e) {}
}

TEST(vector, resize) {
  ustl::vector<int> a{1, 2, 3, 4, 5};

  size_t orig_cap = a.capacity();

  // Resize down. This won't reduce the capacity.
  a.resize(3);
  ASSERT_EQ(3u, a.size());
  ASSERT_EQ(orig_cap, a.capacity());


  a.push_back(4);
  ASSERT_EQ(4u, a.size());

  // Resize up.
  a.resize(5);
  a[4] = 11;

  ASSERT_EQ(4, a[3]);
  ASSERT_EQ(11, a[4]);

  // Resize and fill.
  a.resize(10, -1);
  ASSERT_EQ(-1, a.back());
}

TEST(vector, shrink_to_fit) {
  ustl::vector<int> a(1000);
  const size_t original_capacity = a.capacity();

  a[2] = 10;

  a.shrink_to_fit();
  ASSERT_EQ(original_capacity, a.capacity());

  a.resize(3);
  ASSERT_EQ(10, a[2]);
  ASSERT_TRUE(a.capacity() > a.size());
  a.shrink_to_fit();

  // Check that value remains unchanged.
  ASSERT_EQ(10, a[2]);

  // Check that capacity has shrunk.
  ASSERT_TRUE(a.capacity() < original_capacity);
}


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
