#include "./tests.h"

#include <forward_list>
#include <ustl/forward_list>

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
