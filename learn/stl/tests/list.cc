#include <list>
#include <ustl/list>
#include <ustl/vector>
#include <utility>
#include <vector>

#include "./tests.h"

TEST(std_list, constructors) {
  // default:
  std::list<int> l1;
  l1.insert(l1.begin(), 3);
  ASSERT_EQ(3, l1.front());

  // fill:
  std::list<int> fill1(std::list<int>::size_type(10));
  auto it1 = fill1.begin();
  for (size_t i = 0; i < 10; i++) ASSERT_EQ(*it1++, 0);

  std::list<int> fill2(std::list<int>::size_type(10), static_cast<int>(-1));
  it1 = fill2.begin();
  for (size_t i = 0; i < 10; i++) ASSERT_EQ(*it1++, -1);

  // range:
  std::vector<int> v1{1, 2, 3};
  std::list<int> range1(v1.begin(), v1.end());
  const std::list<int> range1a{1, 2, 3};
  ASSERT_TRUE(range1 == range1a);

  // copy:
  std::list<int> copy1(range1);
  ASSERT_TRUE(copy1 == range1a);

  // initializer list:
  std::list<int> l2{1, 2, 3, 4, 5};
  const std::list<int> l3{1, 2, 3, 4, 5};
  ASSERT_TRUE(l2 == l3);
}

TEST(ustl_list, constructors) {
  // default:
  ustl::list<int> l1;
  l1.insert(l1.begin(), 3);
  ASSERT_EQ(3, l1.front());

  // fill:
  ustl::list<int> fill1(ustl::list<int>::size_type(10));
  auto it1 = fill1.begin();
  for (size_t i = 0; i < 10; i++) ASSERT_EQ(*it1++, 0);

  ustl::list<int> fill2(ustl::list<int>::size_type(10), static_cast<int>(-1));
  auto it2 = fill2.begin();
  for (size_t i = 0; i < 10; i++) ASSERT_EQ(*it2++, -1);

  // range:
  ustl::vector<int> v1{1, 2, 3};
  ustl::list<int> range1(v1.begin(), v1.end());
  const ustl::list<int> range1a{1, 2, 3};
  ASSERT_TRUE(range1 == range1a);

  // copy:
  ustl::list<int> copy1(range1);
  ASSERT_TRUE(copy1 == range1a);

  // initializer list:
  ustl::list<int> l2{1, 2, 3, 4, 5};
  const ustl::list<int> l3{1, 2, 3, 4, 5};
  ASSERT_TRUE(l2 == l3);
}

TEST(std_list, assignment) {
  std::list<int> src{1, 2, 3, 4};
  std::list<int> dst{5, 6, 7};

  dst = src;
  ASSERT_TRUE(src == dst);
}

TEST(ustl_list, assignment) {
  ustl::list<int> src{1, 2, 3, 4};
  ustl::list<int> dst{5, 6, 7};

  dst = src;
  ASSERT_TRUE(src == dst);
}

///////////////
// Capacity: //
///////////////

TEST(std_list_capacity, empty) {
  std::list<int> l1;
  ASSERT_TRUE(l1.empty());
  l1.push_front(1);
  ASSERT_FALSE(l1.empty());
}

TEST(ustl_list_capacity, empty) {
  ustl::list<int> l1;
  ASSERT_TRUE(l1.empty());
  l1.push_front(1);
  ASSERT_FALSE(l1.empty());
}

TEST(std_list_capacity, max_size) {
  std::list<int> l1;
  std::list<char> l2 = {'a', 'b', 'c'};

  ASSERT_GT(l1.max_size(), std::list<int>::allocator_type::size_type(1));
  ASSERT_GT(l2.max_size(), std::list<char>::allocator_type::size_type(3));
}

TEST(ustl_list_capacity, max_size) {
  ustl::list<int> l1;
  ustl::list<char> l2 = {'a', 'b', 'c'};

  ASSERT_GT(l1.max_size(), std::list<int>::allocator_type::size_type(1));
  ASSERT_GT(l2.max_size(), std::list<char>::allocator_type::size_type(3));
}

/////////////////////
// Element access: //
/////////////////////

TEST(std_list, front) {
  std::list<int> l1{1, 2, 3};
  ASSERT_EQ(1, l1.front());
  l1.push_front(-1);
  ASSERT_EQ(-1, l1.front());
}

TEST(ustl_list, front) {
  ustl::list<int> l1{1, 2, 3};
  ASSERT_EQ(1, l1.front());
  l1.push_front(-1);
  ASSERT_EQ(-1, l1.front());
}

////////////////
// Modifiers: //
////////////////

TEST(std_list, emplace_front) {
  std::list<std::pair<int, int>> l1{
      std::pair<int, int>{1, 2},
      std::pair<int, int>{2, 3},
      std::pair<int, int>{3, 4},
  };

  l1.emplace_front(0, 1);

  const std::list<std::pair<int, int>> l2{
      std::pair<int, int>{0, 1},
      std::pair<int, int>{1, 2},
      std::pair<int, int>{2, 3},
      std::pair<int, int>{3, 4},
  };

  ASSERT_TRUE(l1 == l2);
  const std::pair<int, int> v1{0, 1};
  ASSERT_EQ(v1, l2.front());
}

TEST(ustl_list, emplace_front) {
  ustl::list<std::pair<int, int>> l1{
      std::pair<int, int>{1, 2},
      std::pair<int, int>{2, 3},
      std::pair<int, int>{3, 4},
  };

  l1.emplace_front(0, 1);

  const ustl::list<std::pair<int, int>> l2{
      std::pair<int, int>{0, 1},
      std::pair<int, int>{1, 2},
      std::pair<int, int>{2, 3},
      std::pair<int, int>{3, 4},
  };

  ASSERT_TRUE(l1 == l2);
  const std::pair<int, int> v1{0, 1};
  ASSERT_EQ(v1, l2.front());
}

TEST(std_list, push_front) {
  std::list<int> l1{1, 2, 3};
  l1.push_front(-1);
  ASSERT_EQ(-1, l1.front());
}

TEST(ustl_list, push_front) {
  ustl::list<int> l1{1, 2, 3};
  l1.push_front(-1);
  ASSERT_EQ(-1, l1.front());
}

TEST(std_list, pop_front) {
  std::list<int> l1{1, 2, 3};
  l1.pop_front();
  ASSERT_EQ(2, l1.front());
  l1.pop_front();
  ASSERT_EQ(3, l1.front());
  l1.pop_front();
  ASSERT_TRUE(l1.empty());
}

TEST(ustl_list, pop_front) {
  ustl::list<int> l1{1, 2, 3};
  l1.pop_front();
  ASSERT_EQ(2, l1.front());
  l1.pop_front();
  ASSERT_EQ(3, l1.front());
  l1.pop_front();
  ASSERT_TRUE(l1.empty());
}

TEST(std_list, emplace) {
  std::list<std::pair<int, int>> l1{
      std::pair<int, int>{1, 2},
      std::pair<int, int>{2, 3},
      std::pair<int, int>{3, 4},
  };

  l1.emplace(++l1.begin(), 0, 1);

  const std::list<std::pair<int, int>> l2{
      std::pair<int, int>{1, 2},
      std::pair<int, int>{0, 1},
      std::pair<int, int>{2, 3},
      std::pair<int, int>{3, 4},
  };

  ASSERT_TRUE(l1 == l2);
}

TEST(ustl_list, emplace) {
  ustl::list<std::pair<int, int>> l1{
      std::pair<int, int>{1, 2},
      std::pair<int, int>{2, 3},
      std::pair<int, int>{3, 4},
  };

  l1.emplace(++l1.begin(), 0, 1);

  const ustl::list<std::pair<int, int>> l2{
      std::pair<int, int>{1, 2},
      std::pair<int, int>{0, 1},
      std::pair<int, int>{2, 3},
      std::pair<int, int>{3, 4},
  };

  ASSERT_TRUE(l1 == l2);
}

TEST(std_list, insert) {
  std::list<int> l1{1, 2, 3};

  auto r1 = l1.insert(l1.begin(), 0);
  ASSERT_EQ(0, l1.front());
  ASSERT_EQ(*r1, 0);
  ASSERT_EQ(*++r1, 1);

  std::vector<int> v1{4, 5, 6};
  auto r2 = l1.insert(l1.end(), v1.begin(), v1.end());
  ASSERT_EQ(*r2, 4);
  const std::list<int> l2{0, 1, 2, 3, 4, 5, 6};
  ASSERT_TRUE(l1 == l2);

  auto it1 = l1.begin();
  auto r3 = l1.insert(it1, std::list<int>::size_type(2), -1);
  const std::list<int> l3{-1, -1, 0, 1, 2, 3, 4, 5, 6};
  ASSERT_EQ(*r3, -1);
  ASSERT_EQ(*++r3, -1);
  ASSERT_TRUE(l1 == l3);

  std::list<int> l4;
  l4.insert(l4.begin(), l1.begin(), l1.end());
  ASSERT_TRUE(l4 == l1);
}

TEST(ustl_list, insert) {
  ustl::list<int> l1{1, 2, 3};

  auto r1 = l1.insert(l1.begin(), 0);
  ASSERT_EQ(0, l1.front());
  ASSERT_EQ(*r1, 0);
  ASSERT_EQ(*++r1, 1);

  ustl::vector<int> v1{4, 5, 6};
  auto r2 = l1.insert(l1.end(), v1.begin(), v1.end());
  ASSERT_EQ(*r2, 4);
  const ustl::list<int> l2{0, 1, 2, 3, 4, 5, 6};
  ASSERT_TRUE(l1 == l2);

  auto it1 = l1.begin();
  auto r3 = l1.insert(it1, ustl::list<int>::size_type(2), -1);
  const ustl::list<int> l3{-1, -1, 0, 1, 2, 3, 4, 5, 6};
  ASSERT_EQ(*r3, -1);
  ASSERT_EQ(*++r3, -1);
  ASSERT_TRUE(l1 == l3);

  ustl::list<int> l4;
  l4.insert(l4.begin(), l1.begin(), l1.end());
  ASSERT_TRUE(l4 == l1);
}

TEST(std_list, swap) {
  std::list<int> l1{1, 2, 3};
  const std::list<int> l1a{1, 2, 3};

  std::list<int> l2{4, 5, 6, 7};
  const std::list<int> l2a{4, 5, 6, 7};

  l1.swap(l2);
  ASSERT_TRUE(l1 == l2a);
  ASSERT_TRUE(l2 == l1a);

  std::swap(l2, l1);
  ASSERT_TRUE(l1 == l1a);
  ASSERT_TRUE(l2 == l2a);
}

TEST(ustl_list, swap) {
  ustl::list<int> l1{1, 2, 3};
  const ustl::list<int> l1a{1, 2, 3};

  ustl::list<int> l2{4, 5, 6, 7};
  const ustl::list<int> l2a{4, 5, 6, 7};

  l1.swap(l2);
  ASSERT_TRUE(l1 == l2a);
  ASSERT_TRUE(l2 == l1a);

  ustl::swap(l2, l1);  // NOLINT(build/include_what_you_use)
  ASSERT_TRUE(l1 == l1a);
  ASSERT_TRUE(l2 == l2a);
}

TEST(std_list, clear) {
  std::list<int> l1{0, 0, 0, 0};
  const std::list<int> l2{1, 2, 3};

  l1.clear();
  ASSERT_TRUE(l1.begin() == l1.end());

  l1.insert(l1.begin(), l2.begin(), l2.end());
  ASSERT_TRUE(l1 == l2);
}

TEST(ustl_list, clear) {
  ustl::list<int> l1{0, 0, 0, 0};
  const ustl::list<int> l2{1, 2, 3};

  l1.clear();
  ASSERT_TRUE(l1.begin() == l1.end());

  l1.insert(l1.begin(), l2.begin(), l2.end());
  ASSERT_TRUE(l1 == l2);
}

/////////////////
// Operations: //
/////////////////

TEST(std_list, remove) {
  std::list<int> l1{1, 2, 3, 1, 4, 1, 5, 6, 7, 1};
  const std::list<int> l1a{2, 3, 4, 5, 6, 7};

  l1.remove(1);

  ASSERT_TRUE(l1 == l1a);
}

TEST(ustl_list, remove) {
  ustl::list<int> l1{1, 2, 3, 1, 4, 1, 5, 6, 7, 1};
  const ustl::list<int> l1a{2, 3, 4, 5, 6, 7};

  l1.remove(1);

  ASSERT_TRUE(l1 == l1a);
}

TEST(std_list, remove_if) {
  std::list<int> l1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::list<int> l1a{1, 3, 5, 7, 9};

  // remove even numbers
  l1.remove_if([](const int& x) { return !(x % 2); });

  ASSERT_TRUE(l1 == l1a);
}

TEST(ustl_list, remove_if) {
  ustl::list<int> l1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const ustl::list<int> l1a{1, 3, 5, 7, 9};

  // remove even numbers
  l1.remove_if([](const int& x) { return !(x % 2); });

  ASSERT_TRUE(l1 == l1a);
}

TEST(std_list, unique) {
  std::list<int> l1{1, 1, 2, 2, 3, 4, 5, 5};
  const std::list<int> l1a{1, 2, 3, 4, 5};
  const std::list<int> l1b{1, 1, 3, 4, 5};

  l1.unique();
  ASSERT_TRUE(l1 == l1a);

  // binary predicate: *i + *(i-1) == 3
  l1.unique([](const int& x, const int& y) { return x + y == 3; });
}

TEST(ustl_list, unique) {
  ustl::list<int> l1{1, 1, 2, 2, 3, 4, 5, 5};
  const ustl::list<int> l1a{1, 2, 3, 4, 5};
  const ustl::list<int> l1b{1, 1, 3, 4, 5};

  l1.unique();
  ASSERT_TRUE(l1 == l1a);

  // binary predicate: *i + *(i-1) == 3
  l1.unique([](const int& x, const int& y) { return x + y == 3; });
}

TEST(std_list, sort) {
  std::list<int> l1{3, 1, 2, 5, 4};
  const std::list<int> l1a{1, 2, 3, 4, 5};

  l1.sort();
  ASSERT_TRUE(l1 == l1a);
}

TEST(ustl_list, sort) {
  ustl::list<int> l1{3, 1, 2, 5, 4};
  const ustl::list<int> l1a{1, 2, 3, 4, 5};

  l1.sort();
  ASSERT_TRUE(l1 == l1a);
}

TEST(std_list, reverse) {
  std::list<int> l1{1, 2, 3};
  const std::list<int> l1a{3, 2, 1};

  l1.reverse();
  ASSERT_TRUE(l1 == l1a);
}

TEST(ustl_list, reverse) {
  ustl::list<int> l1{1, 2, 3};
  const ustl::list<int> l1a{3, 2, 1};

  l1.reverse();
  ASSERT_TRUE(l1 == l1a);
}

////////////////////////////////////
// Non-member function overloads: //
////////////////////////////////////

// relational ops:

TEST(std_list, relational_ops) {
  const std::list<int> a{1, 2, 3};
  const std::list<int> b{4, 5, 6};
  const std::list<int> c{1, 2, 3};
  const std::list<int> d{1, 2, 3, 4};

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

TEST(ustl_list, relational_ops) {
  const ustl::list<int> a{1, 2, 3};
  const ustl::list<int> b{4, 5, 6};
  const ustl::list<int> c{1, 2, 3};
  const ustl::list<int> d{1, 2, 3, 4};

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

TEST_MAIN();
