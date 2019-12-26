#include <array>
#include <ustl/array>

#include "./tests.h"

///////////////
// Iterators //
///////////////

TEST(std_array_iterators, begin) {
  std::array<int, 5> a{1, 2, 3, 4, 5};

  auto it = a.begin();
  ASSERT_EQ(1, *it++);
  ASSERT_EQ(2, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(4, it[1]);

  const auto cit = a.begin();
  ASSERT_EQ(1, *cit);
}

TEST(ustl_array_iterators, begin) {
  ustl::array<int, 5> a{1, 2, 3, 4, 5};

  auto it = a.begin();
  ASSERT_EQ(1, *it++);
  ASSERT_EQ(2, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(4, it[1]);

  const auto cit = a.begin();
  ASSERT_EQ(1, *cit);
}

TEST(std_array_iterators, rbegin) {
  std::array<int, 5> a{1, 2, 3, 4, 5};

  auto it = a.rbegin();
  ASSERT_EQ(5, *it++);
  ASSERT_EQ(4, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(2, it[1]);
}

TEST(ustl_array_iterators, rbegin) {
  ustl::array<int, 5> a{1, 2, 3, 4, 5};

  auto it = a.rbegin();
  ASSERT_EQ(5, *it++);
  ASSERT_EQ(4, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(2, it[1]);
}

TEST(std_array_iterators, cbegin) {
  std::array<int, 5> a{1, 2, 3, 4, 5};

  auto it = a.cbegin();
  ASSERT_EQ(1, *it++);
  ASSERT_EQ(2, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(4, it[1]);
}

TEST(ustl_array_iterators, cbegin) {
  ustl::array<int, 5> a{1, 2, 3, 4, 5};

  auto it = a.cbegin();
  ASSERT_EQ(1, *it++);
  ASSERT_EQ(2, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(4, it[1]);
}

TEST(std_array_iterators, crbegin) {
  std::array<int, 5> a{1, 2, 3, 4, 5};

  auto it = a.crbegin();
  ASSERT_EQ(5, *it++);
  ASSERT_EQ(4, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(2, it[1]);
}

TEST(ustl_array_iterators, crbegin) {
  ustl::array<int, 5> a{1, 2, 3, 4, 5};

  auto it = a.crbegin();
  ASSERT_EQ(5, *it++);
  ASSERT_EQ(4, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(2, it[1]);
}

TEST(std_array_iterators, end) {
  std::array<int, 5> a{1, 2, 3, 4, 5};
  std::array<int, 5> b;

  auto first = a.begin(), last = a.end();
  auto dst = b.begin();

  while (first != last) *dst++ = *first++;

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(b[i], a[i]);
}

TEST(ustl_array_iterators, end) {
  ustl::array<int, 5> a{1, 2, 3, 4, 5};
  ustl::array<int, 5> b;

  auto first = a.begin(), last = a.end();
  auto dst = b.begin();

  while (first != last) *dst++ = *first++;

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(b[i], a[i]);
}

TEST(std_array_iterators, rend) {
  std::array<int, 5> a{1, 2, 3, 4, 5};
  std::array<int, 5> b{5, 4, 3, 2, 1};
  std::array<int, 5> rev;

  auto first = a.rbegin(), last = a.rend();
  auto dst = rev.begin();

  while (first != last) *dst++ = *first++;

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(rev[i], b[i]);
}

TEST(ustl_array_iterators, rend) {
  ustl::array<int, 5> a{1, 2, 3, 4, 5};
  ustl::array<int, 5> b{5, 4, 3, 2, 1};
  ustl::array<int, 5> rev;

  auto first = a.rbegin(), last = a.rend();
  auto dst = rev.begin();

  while (first != last) *dst++ = *first++;

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(rev[i], b[i]);
}

TEST(std_array_iterators, cend) {
  std::array<int, 5> a{1, 2, 3, 4, 5};
  std::array<int, 5> b;

  auto first = a.cbegin(), last = a.cend();
  auto dst = b.begin();

  while (first != last) *dst++ = *first++;

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(b[i], a[i]);
}

TEST(ustl_array_iterators, cend) {
  ustl::array<int, 5> a{1, 2, 3, 4, 5};
  ustl::array<int, 5> b;

  auto first = a.cbegin(), last = a.cend();
  auto dst = b.begin();

  while (first != last) *dst++ = *first++;

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(b[i], a[i]);
}

TEST(std_array_iterators, crend) {
  std::array<int, 5> a{1, 2, 3, 4, 5};
  std::array<int, 5> b{5, 4, 3, 2, 1};
  std::array<int, 5> rev;

  auto first = a.crbegin(), last = a.crend();
  auto dst = rev.begin();

  while (first != last) *dst++ = *first++;

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(rev[i], b[i]);
}

TEST(ustl_array_iterators, crend) {
  ustl::array<int, 5> a{1, 2, 3, 4, 5};
  ustl::array<int, 5> b{5, 4, 3, 2, 1};
  ustl::array<int, 5> rev;

  auto first = a.crbegin(), last = a.crend();
  auto dst = rev.begin();

  while (first != last) *dst++ = *first++;

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(rev[i], b[i]);
}

//////////////
// Capacity //
//////////////

TEST(std_array, size) {
  std::array<int, 10> a;
  std::array<char, 1> b;
  std::array<int, 100> c;
  std::array<int, 0> d;

  ASSERT_EQ(10u, a.size());
  ASSERT_EQ(1u, b.size());
  ASSERT_EQ(100u, c.size());
  ASSERT_EQ(0u, d.size());
}

TEST(ustl_array, size) {
  ustl::array<int, 10> a;
  ustl::array<char, 1> b;
  ustl::array<int, 100> c;
  ustl::array<int, 0> d;

  ASSERT_EQ(10u, a.size());
  ASSERT_EQ(1u, b.size());
  ASSERT_EQ(100u, c.size());
  ASSERT_EQ(0u, d.size());
}

TEST(std_array, max_size) {
  std::array<int, 10> a;
  std::array<char, 1> b;
  std::array<int, 100> c;
  std::array<int, 0> d;

  ASSERT_EQ(10u, a.max_size());
  ASSERT_EQ(a.size(), a.max_size());
  ASSERT_EQ(1u, b.max_size());
  ASSERT_EQ(b.size(), b.max_size());
  ASSERT_EQ(100u, c.max_size());
  ASSERT_EQ(c.size(), c.max_size());
  ASSERT_EQ(0u, d.max_size());
  ASSERT_EQ(d.size(), d.max_size());
}

TEST(ustl_array, max_size) {
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

TEST(ustl_array, empty) {
  ustl::array<int, 10> a;
  ustl::array<char, 1> b;
  ustl::array<int, 100> c;
  ustl::array<int, 0> d;

  ASSERT_EQ(false, a.empty());
  ASSERT_EQ(false, b.empty());
  ASSERT_EQ(false, c.empty());
  ASSERT_EQ(true, d.empty());
}

TEST(std_array, empty) {
  std::array<int, 10> a;
  std::array<char, 1> b;
  std::array<int, 100> c;
  std::array<int, 0> d;

  ASSERT_EQ(false, a.empty());
  ASSERT_EQ(false, b.empty());
  ASSERT_EQ(false, c.empty());
  ASSERT_EQ(true, d.empty());
}

TEST(ustl_array, subscript) {
  ustl::array<int, 2> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;

  // Test values:
  ASSERT_EQ(1, a[0]);
  ASSERT_EQ(2, a[1]);
}

TEST(std_array, subscript) {
  std::array<int, 2> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;

  // Test values:
  ASSERT_EQ(1, a[0]);
  ASSERT_EQ(2, a[1]);
}

TEST(std_array, at) {
  std::array<int, 2> a;
  std::array<int, 0> b;

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
  } catch (std::out_of_range) {
  }

  try {
    b.at(0) = 1;
    FAIL();
  } catch (std::out_of_range) {
  }
}

TEST(ustl_array, at) {
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
  } catch (std::out_of_range) {
  }

  try {
    b.at(0) = 1;
    FAIL();
  } catch (std::out_of_range) {
  }
}

TEST(std_array, front) {
  const std::array<int, 3> a{1, 2, 3};
  ASSERT_EQ(1, a.front());
}

TEST(ustl_array, front) {
  const ustl::array<int, 3> a{1, 2, 3};
  ASSERT_EQ(1, a.front());
}

TEST(std_array, back) {
  const std::array<int, 3> a{1, 2, 3};
  ASSERT_EQ(3, a.back());
}

TEST(ustl_array, back) {
  const ustl::array<int, 3> a{1, 2, 3};
  ASSERT_EQ(3, a.back());
}

TEST(std_array, data) {
  std::array<int, 3> a;

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

TEST(ustl_array, data) {
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

///////////////////////////////////
// Non-member Function Overloads //
///////////////////////////////////

TEST(std_array, get) {
  std::array<int, 3> a{1, 2, 3};

  ASSERT_EQ(std::get<0>(a), 1);
  ASSERT_EQ(std::get<1>(a), 2);
  ASSERT_EQ(std::get<2>(a), 3);
}

TEST(ustl_array, get) {
  ustl::array<int, 3> a{1, 2, 3};

  ASSERT_EQ(ustl::get<0>(a), 1);
  ASSERT_EQ(ustl::get<1>(a), 2);
  ASSERT_EQ(ustl::get<2>(a), 3);
}

// Relational operators:

TEST(std_array, rational_ops) {
  std::array<int, 3> a{1, 2, 3};
  std::array<int, 3> b{4, 5, 6};
  std::array<int, 3> c{1, 2, 3};

  ASSERT_TRUE(a == c);
  ASSERT_FALSE(a == b);
  ASSERT_TRUE(a != b);
  ASSERT_FALSE(a != c);

  ASSERT_TRUE(a < b);
  ASSERT_FALSE(b < a);
  ASSERT_FALSE(a < c);
  ASSERT_TRUE(a <= c);
  ASSERT_TRUE(a <= b);
  ASSERT_TRUE(b > a);
  ASSERT_TRUE(b >= a);
}

TEST(ustl_array, relational_ops) {
  ustl::array<int, 3> a{1, 2, 3};
  ustl::array<int, 3> b{4, 5, 6};
  ustl::array<int, 3> c{1, 2, 3};

  ASSERT_TRUE(a == c);
  ASSERT_FALSE(a == b);
  ASSERT_TRUE(a != b);
  ASSERT_FALSE(a != c);

  ASSERT_TRUE(a < b);
  ASSERT_FALSE(b < a);
  ASSERT_FALSE(a < c);
  ASSERT_TRUE(a <= c);
  ASSERT_TRUE(a <= b);
  ASSERT_TRUE(b > a);
  ASSERT_TRUE(b >= a);
}

TEST_MAIN();
