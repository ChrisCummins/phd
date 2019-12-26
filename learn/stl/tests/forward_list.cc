#include <forward_list>
#include <ustl/forward_list>
#include <ustl/vector>
#include <utility>
#include <vector>

#include "./tests.h"

TEST(std_forward_list, constructors) {
  // default:
  std::forward_list<int> l1;
  l1.insert_after(l1.before_begin(), 3);
  ASSERT_EQ(3, l1.front());

  // fill:
  std::forward_list<int> fill1(std::forward_list<int>::size_type(10));
  auto it1 = fill1.begin();
  for (size_t i = 0; i < 10; i++) ASSERT_EQ(*it1++, 0);

  std::forward_list<int> fill2(std::forward_list<int>::size_type(10),
                               static_cast<int>(-1));
  it1 = fill2.begin();
  for (size_t i = 0; i < 10; i++) ASSERT_EQ(*it1++, -1);

  // range:
  std::vector<int> v1{1, 2, 3};
  std::forward_list<int> range1(v1.begin(), v1.end());
  const std::forward_list<int> range1a{1, 2, 3};
  ASSERT_TRUE(range1 == range1a);

  // copy:
  std::forward_list<int> copy1(range1);
  ASSERT_TRUE(copy1 == range1a);

  // initializer list:
  std::forward_list<int> l2{1, 2, 3, 4, 5};
  const std::forward_list<int> l3{1, 2, 3, 4, 5};
  ASSERT_TRUE(l2 == l3);
}

TEST(ustl_forward_list, constructors) {
  // default:
  ustl::forward_list<int> l1;
  l1.insert_after(l1.before_begin(), 3);
  ASSERT_EQ(3, l1.front());

  // fill:
  ustl::forward_list<int> fill1(ustl::forward_list<int>::size_type(10));
  auto it1 = fill1.begin();
  for (size_t i = 0; i < 10; i++) ASSERT_EQ(*it1++, 0);

  ustl::forward_list<int> fill2(ustl::forward_list<int>::size_type(10),
                                static_cast<int>(-1));
  auto it2 = fill2.begin();
  for (size_t i = 0; i < 10; i++) ASSERT_EQ(*it2++, -1);

  // range:
  ustl::vector<int> v1{1, 2, 3};
  ustl::forward_list<int> range1(v1.begin(), v1.end());
  const ustl::forward_list<int> range1a{1, 2, 3};
  ASSERT_TRUE(range1 == range1a);

  // copy:
  ustl::forward_list<int> copy1(range1);
  ASSERT_TRUE(copy1 == range1a);

  // initializer list:
  ustl::forward_list<int> l2{1, 2, 3, 4, 5};
  const ustl::forward_list<int> l3{1, 2, 3, 4, 5};
  ASSERT_TRUE(l2 == l3);
}

TEST(std_forward_list, assignment) {
  std::forward_list<int> src{1, 2, 3, 4};
  std::forward_list<int> dst{5, 6, 7};

  dst = src;
  ASSERT_TRUE(src == dst);
}

TEST(ustl_forward_list, assignment) {
  ustl::forward_list<int> src{1, 2, 3, 4};
  ustl::forward_list<int> dst{5, 6, 7};

  dst = src;
  ASSERT_TRUE(src == dst);
}

///////////////
// Capacity: //
///////////////

TEST(std_forward_list, empty) {
  std::forward_list<int> l1;
  ASSERT_TRUE(l1.empty());
  l1.push_front(1);
  ASSERT_FALSE(l1.empty());
}

TEST(ustl_forward_list, empty) {
  ustl::forward_list<int> l1;
  ASSERT_TRUE(l1.empty());
  l1.push_front(1);
  ASSERT_FALSE(l1.empty());
}

TEST(std_forward_list_capacity, max_size) {
  std::forward_list<int> l1;
  std::forward_list<char> l2 = {'a', 'b', 'c'};

  ASSERT_GT(l1.max_size(),
            std::forward_list<int>::allocator_type::size_type(1));
  ASSERT_GT(l2.max_size(),
            std::forward_list<char>::allocator_type::size_type(3));
}

TEST(ustl_forward_list_capacity, max_size) {
  ustl::forward_list<int> l1;
  ustl::forward_list<char> l2 = {'a', 'b', 'c'};

  ASSERT_GT(l1.max_size(),
            std::forward_list<int>::allocator_type::size_type(1));
  ASSERT_GT(l2.max_size(),
            std::forward_list<char>::allocator_type::size_type(3));
}

/////////////////////
// Element access: //
/////////////////////

TEST(std_forward_list, front) {
  std::forward_list<int> l1{1, 2, 3};
  ASSERT_EQ(1, l1.front());

  l1.push_front(-1);
  ASSERT_EQ(-1, l1.front());
}

TEST(ustl_forward_list, front) {
  ustl::forward_list<int> l1{1, 2, 3};
  ASSERT_EQ(1, l1.front());

  l1.push_front(-1);
  ASSERT_EQ(-1, l1.front());
}

////////////////
// Modifiers: //
////////////////

TEST(std_forward_list, emplace_front) {
  std::forward_list<std::pair<int, int>> l1{
      std::pair<int, int>{1, 2},
      std::pair<int, int>{2, 3},
      std::pair<int, int>{3, 4},
  };

  l1.emplace_front(0, 1);

  const std::forward_list<std::pair<int, int>> l2{
      std::pair<int, int>{0, 1},
      std::pair<int, int>{1, 2},
      std::pair<int, int>{2, 3},
      std::pair<int, int>{3, 4},
  };

  ASSERT_TRUE(l1 == l2);
  const std::pair<int, int> v1{0, 1};
  ASSERT_EQ(v1, l2.front());
}

TEST(ustl_forward_list, emplace_front) {
  ustl::forward_list<std::pair<int, int>> l1{
      std::pair<int, int>{1, 2},
      std::pair<int, int>{2, 3},
      std::pair<int, int>{3, 4},
  };

  l1.emplace_front(0, 1);

  const ustl::forward_list<std::pair<int, int>> l2{
      std::pair<int, int>{0, 1},
      std::pair<int, int>{1, 2},
      std::pair<int, int>{2, 3},
      std::pair<int, int>{3, 4},
  };

  ASSERT_TRUE(l1 == l2);
  const std::pair<int, int> v1{0, 1};
  ASSERT_EQ(v1, l2.front());
}

TEST(std_forward_list, push_front) {
  std::forward_list<int> l1{1, 2, 3};
  l1.push_front(-1);
  ASSERT_EQ(-1, l1.front());
}

TEST(ustl_forward_list, push_front) {
  ustl::forward_list<int> l1{1, 2, 3};
  l1.push_front(-1);
  ASSERT_EQ(-1, l1.front());
}

TEST(std_forward_list, pop_front) {
  std::forward_list<int> l1{1, 2, 3};
  l1.pop_front();
  ASSERT_EQ(2, l1.front());
  l1.pop_front();
  ASSERT_EQ(3, l1.front());
  l1.pop_front();
  ASSERT_TRUE(l1.empty());
}

TEST(ustl_forward_list, pop_front) {
  ustl::forward_list<int> l1{1, 2, 3};
  l1.pop_front();
  ASSERT_EQ(2, l1.front());
  l1.pop_front();
  ASSERT_EQ(3, l1.front());
  l1.pop_front();
  ASSERT_TRUE(l1.empty());
}

TEST(std_forward_list, emplace_after) {
  std::forward_list<std::pair<int, int>> l1{
      std::pair<int, int>{1, 2},
      std::pair<int, int>{2, 3},
      std::pair<int, int>{3, 4},
  };

  l1.emplace_after(l1.begin(), 0, 1);

  const std::forward_list<std::pair<int, int>> l2{
      std::pair<int, int>{1, 2},
      std::pair<int, int>{0, 1},
      std::pair<int, int>{2, 3},
      std::pair<int, int>{3, 4},
  };

  ASSERT_TRUE(l1 == l2);
}

TEST(ustl_forward_list, emplace_after) {
  ustl::forward_list<std::pair<int, int>> l1{
      std::pair<int, int>{1, 2},
      std::pair<int, int>{2, 3},
      std::pair<int, int>{3, 4},
  };

  l1.emplace_after(l1.begin(), 0, 1);

  const ustl::forward_list<std::pair<int, int>> l2{
      std::pair<int, int>{1, 2},
      std::pair<int, int>{0, 1},
      std::pair<int, int>{2, 3},
      std::pair<int, int>{3, 4},
  };

  ASSERT_TRUE(l1 == l2);
}

TEST(std_forward_list, insert_after) {
  std::forward_list<int> l1{1, 2, 3};

  auto r1 = l1.insert_after(l1.before_begin(), 0);
  ASSERT_EQ(0, l1.front());
  ASSERT_EQ(*r1, 0);
  ASSERT_EQ(*++r1, 1);

  auto it1 = l1.begin();
  it1++;
  it1++;
  it1++;
  std::vector<int> v1{4, 5, 6};
  auto r2 = l1.insert_after(it1, v1.begin(), v1.end());
  ASSERT_EQ(*r2, 6);
  const std::forward_list<int> l2{0, 1, 2, 3, 4, 5, 6};
  ASSERT_TRUE(l1 == l2);

  it1 = l1.begin();
  auto r3 = l1.insert_after(it1, std::forward_list<int>::size_type(2), -1);
  const std::forward_list<int> l3{0, -1, -1, 1, 2, 3, 4, 5, 6};
  ASSERT_EQ(*r3, -1);
  ASSERT_EQ(*++r3, 1);
  ASSERT_TRUE(l1 == l3);

  std::forward_list<int> l4;
  l4.insert_after(l4.before_begin(), l1.begin(), l1.end());
  ASSERT_TRUE(l4 == l1);
}

TEST(ustl_forward_list, insert_after) {
  ustl::forward_list<int> l1{1, 2, 3};

  auto r1 = l1.insert_after(l1.before_begin(), 0);
  ASSERT_EQ(0, l1.front());
  ASSERT_EQ(*r1, 0);
  ASSERT_EQ(*++r1, 1);

  auto it1 = l1.begin();
  it1++;
  it1++;
  it1++;
  ustl::vector<int> v1{4, 5, 6};
  auto r2 = l1.insert_after(it1, v1.begin(), v1.end());
  ASSERT_EQ(*r2, 6);
  const ustl::forward_list<int> l2{0, 1, 2, 3, 4, 5, 6};
  ASSERT_TRUE(l1 == l2);

  it1 = l1.begin();
  auto r3 = l1.insert_after(it1, ustl::forward_list<int>::size_type(2), -1);
  const ustl::forward_list<int> l3{0, -1, -1, 1, 2, 3, 4, 5, 6};
  ASSERT_EQ(*r3, -1);
  ASSERT_EQ(*++r3, 1);
  ASSERT_TRUE(l1 == l3);

  ustl::forward_list<int> l4;
  l4.insert_after(l4.before_begin(), l1.begin(), l1.end());
  ASSERT_TRUE(l4 == l1);
}

TEST(std_forward_list, swap) {
  std::forward_list<int> l1{1, 2, 3};
  const std::forward_list<int> l1a{1, 2, 3};

  std::forward_list<int> l2{4, 5, 6, 7};
  const std::forward_list<int> l2a{4, 5, 6, 7};

  l1.swap(l2);
  ASSERT_TRUE(l1 == l2a);
  ASSERT_TRUE(l2 == l1a);

  std::swap(l2, l1);
  ASSERT_TRUE(l1 == l1a);
  ASSERT_TRUE(l2 == l2a);
}

TEST(ustl_forward_list, swap) {
  ustl::forward_list<int> l1{1, 2, 3};
  const ustl::forward_list<int> l1a{1, 2, 3};

  ustl::forward_list<int> l2{4, 5, 6, 7};
  const ustl::forward_list<int> l2a{4, 5, 6, 7};

  l1.swap(l2);
  ASSERT_TRUE(l1 == l2a);
  ASSERT_TRUE(l2 == l1a);

  ustl::swap(l2, l1);  // NOLINT(build/include_what_you_use)
  ASSERT_TRUE(l1 == l1a);
  ASSERT_TRUE(l2 == l2a);
}

TEST(std_forward_list, clear) {
  std::forward_list<int> l1{0, 0, 0, 0};
  const std::forward_list<int> l2{1, 2, 3};

  l1.clear();
  ASSERT_TRUE(l1.begin() == l1.end());

  l1.insert_after(l1.before_begin(), l2.begin(), l2.end());
  ASSERT_TRUE(l1 == l2);
}

TEST(ustl_forward_list, clear) {
  ustl::forward_list<int> l1{0, 0, 0, 0};
  const ustl::forward_list<int> l2{1, 2, 3};

  l1.clear();
  ASSERT_TRUE(l1.begin() == l1.end());

  l1.insert_after(l1.before_begin(), l2.begin(), l2.end());
  ASSERT_TRUE(l1 == l2);
}

/////////////////
// Operations: //
/////////////////

TEST(std_forward_list, remove) {
  std::forward_list<int> l1{1, 2, 3, 1, 4, 1, 5, 6, 7, 1};
  const std::forward_list<int> l1a{2, 3, 4, 5, 6, 7};

  l1.remove(1);

  ASSERT_TRUE(l1 == l1a);
}

TEST(ustl_forward_list, remove) {
  ustl::forward_list<int> l1{1, 2, 3, 1, 4, 1, 5, 6, 7, 1};
  const ustl::forward_list<int> l1a{2, 3, 4, 5, 6, 7};

  l1.remove(1);

  ASSERT_TRUE(l1 == l1a);
}

TEST(std_forward_list, remove_if) {
  std::forward_list<int> l1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::forward_list<int> l1a{1, 3, 5, 7, 9};

  // remove even numbers
  l1.remove_if([](const int& x) { return !(x % 2); });

  ASSERT_TRUE(l1 == l1a);
}

TEST(ustl_forward_list, remove_if) {
  ustl::forward_list<int> l1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const ustl::forward_list<int> l1a{1, 3, 5, 7, 9};

  // remove even numbers
  l1.remove_if([](const int& x) { return !(x % 2); });

  ASSERT_TRUE(l1 == l1a);
}

TEST(std_forward_list, unique) {
  std::forward_list<int> l1{1, 1, 2, 2, 3, 4, 5, 5};
  const std::forward_list<int> l1a{1, 2, 3, 4, 5};
  const std::forward_list<int> l1b{1, 1, 3, 4, 5};

  l1.unique();
  ASSERT_TRUE(l1 == l1a);

  // binary predicate: *i + *(i-1) == 3
  l1.unique([](const int& x, const int& y) { return x + y == 3; });
}

TEST(ustl_forward_list, unique) {
  ustl::forward_list<int> l1{1, 1, 2, 2, 3, 4, 5, 5};
  const ustl::forward_list<int> l1a{1, 2, 3, 4, 5};
  const ustl::forward_list<int> l1b{1, 1, 3, 4, 5};

  l1.unique();
  ASSERT_TRUE(l1 == l1a);

  // binary predicate: *i + *(i-1) == 3
  l1.unique([](const int& x, const int& y) { return x + y == 3; });
}

TEST(std_forward_list, sort) {
  std::forward_list<int> l1{3, 1, 2, 5, 4};
  const std::forward_list<int> l1a{1, 2, 3, 4, 5};

  l1.sort();
  ASSERT_TRUE(l1 == l1a);
}

TEST(ustl_forward_list, sort) {
  ustl::forward_list<int> l1{3, 1, 2, 5, 4};
  const ustl::forward_list<int> l1a{1, 2, 3, 4, 5};

  l1.sort();
  ASSERT_TRUE(l1 == l1a);
}

TEST(std_forward_list, reverse) {
  std::forward_list<int> l1{1, 2, 3};
  const std::forward_list<int> l1a{3, 2, 1};

  l1.reverse();
  ASSERT_TRUE(l1 == l1a);
}

TEST(ustl_forward_list, reverse) {
  ustl::forward_list<int> l1{1, 2, 3};
  const ustl::forward_list<int> l1a{3, 2, 1};

  l1.reverse();
  ASSERT_TRUE(l1 == l1a);
}

////////////////////////////////////
// Non-member function overloads: //
////////////////////////////////////

// relational ops:

TEST(std_forward_list, relational_ops) {
  const std::forward_list<int> a{1, 2, 3};
  const std::forward_list<int> b{4, 5, 6};
  const std::forward_list<int> c{1, 2, 3};
  const std::forward_list<int> d{1, 2, 3, 4};

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

TEST(ustl_forward_list, relational_ops) {
  const ustl::forward_list<int> a{1, 2, 3};
  const ustl::forward_list<int> b{4, 5, 6};
  const ustl::forward_list<int> c{1, 2, 3};
  const ustl::forward_list<int> d{1, 2, 3, 4};

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
