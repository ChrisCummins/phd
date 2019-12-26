#include <algorithm>
#include <ustl/vector>
#include <vector>

#include "./tests.h"

TEST(std_vector, constructors) {
  std::vector<int> a(3);
  std::vector<double> b(3, 3.5f);
  std::vector<char> c = {'a', 'b', 'c'};

  for (size_t i = 0; i < 3; i++) ASSERT_EQ(0, a[i]);

  for (size_t i = 0; i < 3; i++) ASSERT_EQ(3.5f, b[i]);

  ASSERT_EQ('a', c[0]);
  ASSERT_EQ('b', c[1]);
  ASSERT_EQ('c', c[2]);

  // Range constructor;
  std::vector<char> d(c.begin(), c.end());
  ASSERT_TRUE(c == d);

  // Copy constructor:
  std::vector<char> e = d;
}

TEST(ustl_vector, constructors) {
  ustl::vector<int> a(3);
  ustl::vector<double> b(3, 3.5f);
  ustl::vector<char> c = {'a', 'b', 'c'};

  for (size_t i = 0; i < 3; i++) ASSERT_EQ(0, a[i]);

  for (size_t i = 0; i < 3; i++) ASSERT_EQ(3.5f, b[i]);

  ASSERT_EQ('a', c[0]);
  ASSERT_EQ('b', c[1]);
  ASSERT_EQ('c', c[2]);

  // Range constructor;
  ustl::vector<char> d(c.begin(), c.end());
  ASSERT_TRUE(c == d);

  // Copy constructor:
  ustl::vector<char> e = d;
}

//////////////
// Capacity //
//////////////

TEST(std_vector_capacity, size) {
  std::vector<int> a;
  std::vector<int> b(5);
  std::vector<double> c(6, 3.5f);
  std::vector<char> d = {'a', 'b', 'c'};

  ASSERT_EQ(0u, a.size());
  ASSERT_EQ(5u, b.size());
  ASSERT_EQ(6u, c.size());
  ASSERT_EQ(3u, d.size());
}

TEST(ustl_vector_capacity, size) {
  ustl::vector<int> a;
  ustl::vector<int> b(5);
  ustl::vector<double> c(6, 3.5f);
  ustl::vector<char> d = {'a', 'b', 'c'};

  ASSERT_EQ(0u, a.size());
  ASSERT_EQ(5u, b.size());
  ASSERT_EQ(6u, c.size());
  ASSERT_EQ(3u, d.size());
}

TEST(std_vector_capacity, max_size) {
  std::vector<int> a;
  std::vector<int> b(5);
  std::vector<double> c(6, 3.5f);
  std::vector<char> d = {'a', 'b', 'c'};

  ASSERT_TRUE(a.size() < a.max_size());
  ASSERT_TRUE(b.size() < b.max_size());
  ASSERT_TRUE(c.size() < c.max_size());

  ASSERT_EQ(a.max_size(), b.max_size());
}

TEST(ustl_vector_capacity, max_size) {
  ustl::vector<int> a;
  ustl::vector<int> b(5);
  ustl::vector<double> c(6, 3.5f);
  ustl::vector<char> d = {'a', 'b', 'c'};

  ASSERT_TRUE(a.size() < a.max_size());
  ASSERT_TRUE(b.size() < b.max_size());
  ASSERT_TRUE(c.size() < c.max_size());

  ASSERT_EQ(a.max_size(), b.max_size());
}

TEST(std_vector_capacity, resize) {
  std::vector<int> a{1, 2, 3, 4, 5};

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

TEST(ustl_vector_capacity, resize) {
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

TEST(std_vector_capacity, capacity) {
  std::vector<int> a;
  std::vector<int> b(5);
  std::vector<double> c(6, 3.5f);
  std::vector<char> d = {'a', 'b', 'c'};

  ASSERT_TRUE(a.capacity() >= a.size());
  ASSERT_TRUE(b.capacity() >= b.size());
  ASSERT_TRUE(c.capacity() >= c.size());
  ASSERT_TRUE(d.capacity() >= d.size());
}

TEST(ustl_vector_capacity, capacity) {
  ustl::vector<int> a;
  ustl::vector<int> b(5);
  ustl::vector<double> c(6, 3.5f);
  ustl::vector<char> d = {'a', 'b', 'c'};

  ASSERT_TRUE(a.capacity() >= a.size());
  ASSERT_TRUE(b.capacity() >= b.size());
  ASSERT_TRUE(c.capacity() >= c.size());
  ASSERT_TRUE(d.capacity() >= d.size());
}

TEST(std_vector_capacity, empty) {
  std::vector<int> a;
  std::vector<int> b(0);
  std::vector<int> c{1, 2, 3};

  ASSERT_TRUE(a.empty());
  ASSERT_TRUE(b.empty());
  ASSERT_FALSE(c.empty());
}

TEST(ustl_vector_capacity, empty) {
  ustl::vector<int> a;
  ustl::vector<int> b(0);
  ustl::vector<int> c{1, 2, 3};

  ASSERT_TRUE(a.empty());
  ASSERT_TRUE(b.empty());
  ASSERT_FALSE(c.empty());
}

TEST(std_vector_capacity, reserve) {
  std::vector<int> a(100);
  const size_t original_capacity = a.capacity();

  a.reserve(3);
  ASSERT_EQ(a.capacity(), original_capacity);

  a.reserve(10000);
  ASSERT_TRUE(a.capacity() > original_capacity);
}

TEST(ustl_vector_capacity, reserve) {
  ustl::vector<int> a(100);
  const size_t original_capacity = a.capacity();

  a.reserve(3);
  ASSERT_EQ(a.capacity(), original_capacity);

  a.reserve(10000);
  ASSERT_TRUE(a.capacity() > original_capacity);
}

TEST(std_vector_capacity, shrink_to_fit) {
  std::vector<int> a(1000);
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

TEST(ustl_vector_capacity, shrink_to_fit) {
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

///////////////
// Iterators //
///////////////

TEST(std_vector_iterators, begin) {
  std::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.begin();
  ASSERT_EQ(1, *it++);
  ASSERT_EQ(2, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(4, it[1]);

  const auto cit = a.begin();
  ASSERT_EQ(1, *cit);
}

TEST(ustl_vector_iterators, begin) {
  ustl::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.begin();
  ASSERT_EQ(1, *it++);
  ASSERT_EQ(2, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(4, it[1]);

  const auto cit = a.begin();
  ASSERT_EQ(1, *cit);
}

TEST(std_vector_iterators, rbegin) {
  std::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.rbegin();
  ASSERT_EQ(5, *it++);
  ASSERT_EQ(4, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(2, it[1]);
}

TEST(ustl_vector_iterators, rbegin) {
  ustl::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.rbegin();
  ASSERT_EQ(5, *it++);
  ASSERT_EQ(4, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(2, it[1]);
}

TEST(std_vector_iterators, cbegin) {
  std::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.cbegin();
  ASSERT_EQ(1, *it++);
  ASSERT_EQ(2, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(4, it[1]);
}

TEST(ustl_vector_iterators, cbegin) {
  ustl::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.cbegin();
  ASSERT_EQ(1, *it++);
  ASSERT_EQ(2, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(4, it[1]);
}

TEST(std_vector_iterators, crbegin) {
  std::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.crbegin();
  ASSERT_EQ(5, *it++);
  ASSERT_EQ(4, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(2, it[1]);
}

TEST(ustl_vector_iterators, crbegin) {
  ustl::vector<int> a{1, 2, 3, 4, 5};

  auto it = a.crbegin();
  ASSERT_EQ(5, *it++);
  ASSERT_EQ(4, *it++);
  ASSERT_EQ(3, *it);
  ASSERT_EQ(2, it[1]);
}

TEST(std_vector_iterators, end) {
  std::vector<int> a{1, 2, 3, 4, 5};
  std::vector<int> b;

  auto first = a.begin(), last = a.end();

  while (first != last) b.push_back(*first++);

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(b[i], a[i]);
}

TEST(ustl_vector_iterators, end) {
  ustl::vector<int> a{1, 2, 3, 4, 5};
  ustl::vector<int> b;

  auto first = a.begin(), last = a.end();

  while (first != last) b.push_back(*first++);

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(b[i], a[i]);
}

TEST(std_vector_iterators, rend) {
  std::vector<int> a{1, 2, 3, 4, 5};
  std::vector<int> b{5, 4, 3, 2, 1};
  std::vector<int> rev;

  auto first = a.rbegin(), last = a.rend();

  while (first != last) rev.push_back(*first++);

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(rev[i], b[i]);
}

TEST(ustl_vector_iterators, rend) {
  ustl::vector<int> a{1, 2, 3, 4, 5};
  ustl::vector<int> b{5, 4, 3, 2, 1};
  ustl::vector<int> rev;

  auto first = a.rbegin(), last = a.rend();

  while (first != last) rev.push_back(*first++);

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(rev[i], b[i]);
}

TEST(std_vector_iterators, cend) {
  std::vector<int> a{1, 2, 3, 4, 5};
  std::vector<int> b;

  auto first = a.cbegin(), last = a.cend();

  while (first != last) b.push_back(*first++);

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(b[i], a[i]);
}

TEST(ustl_vector_iterators, cend) {
  ustl::vector<int> a{1, 2, 3, 4, 5};
  ustl::vector<int> b;

  auto first = a.cbegin(), last = a.cend();

  while (first != last) b.push_back(*first++);

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(b[i], a[i]);
}

TEST(std_vector_iterators, crend) {
  std::vector<int> a{1, 2, 3, 4, 5};
  std::vector<int> b{5, 4, 3, 2, 1};
  std::vector<int> rev;

  auto first = a.crbegin(), last = a.crend();

  while (first != last) rev.push_back(*first++);

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(rev[i], b[i]);
}

TEST(ustl_vector_iterators, crend) {
  ustl::vector<int> a{1, 2, 3, 4, 5};
  ustl::vector<int> b{5, 4, 3, 2, 1};
  ustl::vector<int> rev;

  auto first = a.crbegin(), last = a.crend();

  while (first != last) rev.push_back(*first++);

  for (size_t i = 0; i < 5; i++) ASSERT_EQ(rev[i], b[i]);
}

////////////////////
// Element access //
////////////////////

TEST(std_vector_access, front) {
  std::vector<int> a(3);
  std::vector<double> b(3, 3.5f);
  std::vector<char> c = {'a', 'b', 'c'};

  ASSERT_EQ(0, a.front());
  ASSERT_EQ(3.5, b.front());
  ASSERT_EQ('a', c.front());
}

TEST(ustl_vector_access, front) {
  ustl::vector<int> a(3);
  ustl::vector<double> b(3, 3.5f);
  ustl::vector<char> c = {'a', 'b', 'c'};

  ASSERT_EQ(0, a.front());
  ASSERT_EQ(3.5, b.front());
  ASSERT_EQ('a', c.front());
}

TEST(std_vector_access, back) {
  std::vector<int> a(3);
  std::vector<double> b(3, 3.5f);
  std::vector<char> c = {'a', 'b', 'c'};

  ASSERT_EQ(0, a.back());
  ASSERT_EQ(3.5, b.back());
  ASSERT_EQ('c', c.back());
}

TEST(ustl_vector_access, back) {
  ustl::vector<int> a(3);
  ustl::vector<double> b(3, 3.5f);
  ustl::vector<char> c = {'a', 'b', 'c'};

  ASSERT_EQ(0, a.back());
  ASSERT_EQ(3.5, b.back());
  ASSERT_EQ('c', c.back());
}

TEST(std_vector_access, at) {
  std::vector<int> a(3);
  std::vector<double> b(3, 3.5f);
  std::vector<char> c = {'a', 'b', 'c'};

  for (size_t i = 0; i < 3; i++) {
    ASSERT_EQ(0, a.at(i));
    ASSERT_EQ(a[i], a.at(i));
  }
  try {
    a.at(3);
    FAIL();
  } catch (std::out_of_range&) {
  }

  for (size_t i = 0; i < 3; i++) {
    ASSERT_EQ(3.5f, b.at(i));
    ASSERT_EQ(b[i], b.at(i));
  }
  try {
    b.at(3);
    FAIL();
  } catch (std::out_of_range&) {
  }

  ASSERT_EQ('a', c.at(0));
  ASSERT_EQ('b', c.at(1));
  ASSERT_EQ('c', c.at(2));
  try {
    c.at(3);
    FAIL();
  } catch (std::out_of_range&) {
  }
}

TEST(ustl_vector_access, at) {
  ustl::vector<int> a(3);
  ustl::vector<double> b(3, 3.5f);
  ustl::vector<char> c = {'a', 'b', 'c'};

  for (size_t i = 0; i < 3; i++) {
    ASSERT_EQ(0, a.at(i));
    ASSERT_EQ(a[i], a.at(i));
  }
  try {
    a.at(3);
    FAIL();
  } catch (std::out_of_range&) {
  }

  for (size_t i = 0; i < 3; i++) {
    ASSERT_EQ(3.5f, b.at(i));
    ASSERT_EQ(b[i], b.at(i));
  }
  try {
    b.at(3);
    FAIL();
  } catch (std::out_of_range&) {
  }

  ASSERT_EQ('a', c.at(0));
  ASSERT_EQ('b', c.at(1));
  ASSERT_EQ('c', c.at(2));
  try {
    c.at(3);
    FAIL();
  } catch (std::out_of_range&) {
  }
}

///////////////
// Modifiers //
///////////////

TEST(std_vector_modifiers, assign) {
  std::vector<int> a;
  std::vector<int> b{1, 2, 3};

  a.assign(b.begin(), b.end());
  for (size_t i = 0; i < 3; i++) ASSERT_EQ(a[i], b[i]);

  a.assign(std::vector<int>::size_type(100), 3);
  for (size_t i = 0; i < 100; i++) ASSERT_EQ(a[i], 3);

  a.assign({0, 1, 2, 3});
  for (size_t i = 0; i < 4; i++) ASSERT_EQ(a[i], static_cast<int>(i));

  ASSERT_EQ(a.size(), std::vector<int>::size_type(4));
}

TEST(ustl_vector_modifiers, assign) {
  ustl::vector<int> a;
  ustl::vector<int> b{1, 2, 3};

  a.assign(b.begin(), b.end());
  for (size_t i = 0; i < 3; i++) ASSERT_EQ(a[i], b[i]);

  a.assign(ustl::vector<int>::size_type(100), 3);
  for (size_t i = 0; i < 100; i++) ASSERT_EQ(a[i], 3);

  a.assign({0, 1, 2, 3});
  for (size_t i = 0; i < 4; i++) ASSERT_EQ(a[i], static_cast<int>(i));

  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(4));
}

TEST(std_vector_modifiers, push_back) {
  std::vector<int> a;

  a.push_back(1);
  a.push_back(2);

  ASSERT_EQ(a.size(), std::vector<int>::size_type(2));
  ASSERT_EQ(a[0], 1);
  ASSERT_EQ(a[1], 2);
}

TEST(ustl_vector_modifiers, push_back) {
  ustl::vector<int> a;

  a.push_back(1);
  a.push_back(2);

  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(2));
  ASSERT_EQ(a[0], 1);
  ASSERT_EQ(a[1], 2);
}

TEST(std_vector_modifiers, pop_back) {
  std::vector<int> a{1, 2, 3};

  a.pop_back();
  a.pop_back();
  ASSERT_EQ(a.size(), std::vector<int>::size_type(1));
  ASSERT_EQ(a[0], 1);
}

TEST(ustl_vector_modifiers, pop_back) {
  ustl::vector<int> a{1, 2, 3};

  a.pop_back();
  a.pop_back();
  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(1));
  ASSERT_EQ(a[0], 1);
}

TEST(std_vector_modifiers, insert) {
  std::vector<int> a{1, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> b{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> c{1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> d{-1, -2, -3, 1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> e{-1, -2, -3, 1, 2, 3, 0,  0,  0,
                     4,  5,  6,  7, 8, 9, -1, -2, 10};

  std::vector<int> ins{-1, -2, -3};

  a.insert(a.begin() + 1, 2);
  ASSERT_EQ(a.size(), std::vector<int>::size_type(10));
  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], b[i]);

  a.insert(a.begin() + 3, std::vector<int>::size_type(3), static_cast<int>(0));
  ASSERT_EQ(a.size(), std::vector<int>::size_type(13));
  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], c[i]);

  a.insert(a.begin(), ins.begin(), ins.end());
  ASSERT_EQ(a.size(), std::vector<int>::size_type(16));
  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], d[i]);

  a.insert(a.end() - 1, {-1, -2});
  ASSERT_EQ(a.size(), std::vector<int>::size_type(18));
  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], e[i]);
}

TEST(ustl_vector_modifiers, insert) {
  ustl::vector<int> a{1, 3, 4, 5, 6, 7, 8, 9, 10};
  ustl::vector<int> b{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  ustl::vector<int> c{1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10};
  ustl::vector<int> d{-1, -2, -3, 1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10};
  ustl::vector<int> e{-1, -2, -3, 1, 2, 3, 0,  0,  0,
                      4,  5,  6,  7, 8, 9, -1, -2, 10};

  ustl::vector<int> ins{-1, -2, -3};

  a.insert(a.begin() + 1, 2);
  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(10));
  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], b[i]);

  a.insert(a.begin() + 3, ustl::vector<int>::size_type(3), static_cast<int>(0));
  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(13));
  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], c[i]);

  a.insert(a.begin(), ins.begin(), ins.end());
  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(16));
  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], d[i]);

  a.insert(a.end() - 1, {-1, -2});
  ASSERT_EQ(a.size(), ustl::vector<int>::size_type(18));
  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], e[i]);
}

TEST(std_vector_modifiers, erase) {
  std::vector<int> v1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::vector<int> v1a{1, 3, 4, 5, 6, 8, 9, 10};
  v1.erase(v1.begin() + 6);
  v1.erase(v1.begin() + 1);
  ASSERT_TRUE(v1 == v1a);

  std::vector<int> v2{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::vector<int> v2a{1, 10};
  v2.erase(v2.begin() + 1, v2.end() - 1);
  ASSERT_TRUE(v2 == v2a);
}

TEST(ustl_vector_modifiers, erase) {
  ustl::vector<int> v1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const ustl::vector<int> v1a{1, 3, 4, 5, 6, 8, 9, 10};
  v1.erase(v1.begin() + 6);
  v1.erase(v1.begin() + 1);
  ASSERT_TRUE(v1 == v1a);

  ustl::vector<int> v2{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const ustl::vector<int> v2a{1, 10};
  v2.erase(v2.begin() + 1, v2.end() - 1);
  ASSERT_TRUE(v2 == v2a);
}

TEST(std_vector_modifiers, swap) {
  std::vector<int> a{1, 2, 3};
  std::vector<int> b{4, 5};

  a.swap(b);

  ASSERT_EQ(std::vector<int>::size_type(2), a.size());
  ASSERT_EQ(std::vector<int>::size_type(3), b.size());
  ASSERT_TRUE(vector_equal(a, {4, 5}));
  ASSERT_TRUE(vector_equal(b, {1, 2, 3}));
}

TEST(ustl_vector_modifiers, swap) {
  ustl::vector<int> a{1, 2, 3};
  ustl::vector<int> b{4, 5};

  a.swap(b);

  ASSERT_EQ(ustl::vector<int>::size_type(2), a.size());
  ASSERT_EQ(ustl::vector<int>::size_type(3), b.size());
  vector_equal(a, {4, 5});
  vector_equal(b, {1, 2, 3});
}

TEST(std_vector_modifiers, clear) {
  std::vector<int> v1{1, 2, 3};
  v1.clear();
  ASSERT_TRUE(v1.empty());
}

TEST(ustl_vector_modifiers, clear) {
  ustl::vector<int> v1{1, 2, 3};
  v1.clear();
  ASSERT_TRUE(v1.empty());
}

TEST(std_vector_modifiers, emplace) {
  std::vector<int> v1{2, 3};
  const std::vector<int> v1a{1, 2, 3};
  v1.emplace(v1.begin(), 1);
  ASSERT_TRUE(v1 == v1a);
}

TEST(ustl_vector_modifiers, emplace) {
  ustl::vector<int> v1{2, 3};
  const ustl::vector<int> v1a{1, 2, 3};
  v1.emplace(v1.begin(), 1);
  ASSERT_TRUE(v1 == v1a);
}

TEST(std_vector_modifiers, emplace_back) {
  std::vector<int> v1{1, 2};
  const std::vector<int> v1a{1, 2, 3, 4};
  v1.emplace_back(3);
  v1.emplace_back(4);
  ASSERT_TRUE(v1 == v1a);
}

TEST(ustl_vector_modifiers, emplace_back) {
  ustl::vector<int> v1{1, 2};
  const ustl::vector<int> v1a{1, 2, 3, 4};
  v1.emplace_back(3);
  v1.emplace_back(4);
  ASSERT_TRUE(v1 == v1a);
}

////////////////////////////////////
// Non-member function overloads: //
////////////////////////////////////

// relational operators

TEST(std_vector, rational_ops) {
  std::vector<int> a{1, 2, 3};
  std::vector<int> b{4, 5, 6};
  std::vector<int> c{1, 2, 3};
  std::vector<int> d{1, 2, 3, 4};

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

  ASSERT_TRUE(a < d);
  ASSERT_FALSE(d >= b);
  ASSERT_TRUE(d >= a);
  ASSERT_TRUE(d > a);
}

TEST(ustl_vector, relational_ops) {
  ustl::vector<int> a{1, 2, 3};
  ustl::vector<int> b{4, 5, 6};
  ustl::vector<int> c{1, 2, 3};
  ustl::vector<int> d{1, 2, 3, 4};

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

  ASSERT_TRUE(a < d);
  ASSERT_FALSE(d >= b);
  ASSERT_TRUE(d >= a);
  ASSERT_TRUE(d > a);
}

TEST(std_vector_modifiers, swap_overload) {
  std::vector<int> a{1, 2, 3};
  std::vector<int> b{4, 5};

  std::swap(a, b);

  ASSERT_EQ(std::vector<int>::size_type(2), a.size());
  ASSERT_EQ(std::vector<int>::size_type(3), b.size());
  vector_equal(a, {4, 5});
  vector_equal(b, {1, 2, 3});
}

TEST(ustl_vector_modifiers, swap_overload) {
  ustl::vector<int> a{1, 2, 3};
  ustl::vector<int> b{4, 5};

  ustl::swap(a, b);

  ASSERT_EQ(ustl::vector<int>::size_type(2), a.size());
  ASSERT_EQ(ustl::vector<int>::size_type(3), b.size());
  vector_equal(a, {4, 5});
  vector_equal(b, {1, 2, 3});
}

TEST_MAIN();
