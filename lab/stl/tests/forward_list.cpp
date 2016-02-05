#include "./tests.h"

#include <forward_list>
#include <ustl/forward_list>

#include <vector>
#include <ustl/vector>

#include <utility>

TEST(std_forward_list, constructors) {
  std::forward_list<int> l1;
  l1.insert_after(l1.before_begin(), 3);
  ASSERT_EQ(3, l1.front());

  std::forward_list<int> l2{1, 2, 3, 4, 5};
  std::forward_list<int> l3{1, 2, 3, 4, 5};

  ASSERT_TRUE(l2 == l3);
}


TEST(ustl_forward_list, constructors) {
  ustl::forward_list<int> l1;
  l1.insert_after(l1.before_begin(), 3);
  ASSERT_EQ(3, l1.front());

  ustl::forward_list<int> l2{1, 2, 3, 4, 5};
  ustl::forward_list<int> l3{1, 2, 3, 4, 5};

  ASSERT_TRUE(l2 == l3);
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
  std::forward_list<char> l2 = { 'a', 'b', 'c' };

  ASSERT_GT(l1.max_size(),
            std::forward_list<int>::allocator_type::size_type(1));
  ASSERT_GT(l2.max_size(),
            std::forward_list<char>::allocator_type::size_type(3));
}

TEST(ustl_forward_list_capacity, max_size) {
  ustl::forward_list<int> l1;
  ustl::forward_list<char> l2 = { 'a', 'b', 'c' };

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


TEST(std_forward_list, insert_after) {
  std::forward_list<int> l1{1, 2, 3};

  auto r1 = l1.insert_after(l1.before_begin(), 0);
  ASSERT_EQ(0, l1.front());
  ASSERT_EQ(*r1, 0);
  ASSERT_EQ(*++r1, 1);

  auto it1 = l1.begin();
  it1++; it1++; it1++;
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
}

TEST(ustl_forward_list, insert_after) {
  ustl::forward_list<int> l1{1, 2, 3};

  auto r1 = l1.insert_after(l1.before_begin(), 0);
  ASSERT_EQ(0, l1.front());
  ASSERT_EQ(*r1, 0);
  ASSERT_EQ(*++r1, 1);

  auto it1 = l1.begin();
  it1++; it1++; it1++;
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
