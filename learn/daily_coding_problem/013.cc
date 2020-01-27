// This problem was asked by Twitter.
//
// You run an e-commerce website and want to record the last N order ids in a
// log. Implement a data structure to accomplish this, with the following API:
//
// record(order_id): adds the order_id to the log
// get_last(i): gets the ith last element from the log. i is guaranteed to be
// smaller than or equal to N.
// You should be as efficient with time and space as possible.

#include <array>
#include <vector>
#include "labm8/cpp/test.h"

using std::array;
using std::min;
using std::vector;

template <size_t N>
class R {
 public:
  R() : s_(0), c_(0) {}

  // Time: O(i)
  // Space: O(i)
  vector<int> get_last(int i) const {
    vector<int> v;
    int s = min(s_, i);
    v.reserve(s);

    for (int j = c_; j > c_ - s; --j) {
      v.push_back(v_[j % N]);
    }

    return v;
  }

  // Same as above, but templated method to return fixed size array.
  template <size_t I>
  array<int, I> get_last() const {
    array<int, I> v;

    for (int i = 0; i < v.size(); ++i) {
      v[i] = v_[(c_ - i) % N];
    }

    return v;
  }

  // Time: O(1)
  // Space: O(1)
  void record(int order_id) {
    c_ = (c_ + 1) % N;
    v_[c_] = order_id;
    ++s_;
  }

 private:
  array<int, N> v_;
  int s_;
  int c_;
};

TEST(RollingLog, Empty) {
  R<5> r;
  EXPECT_EQ(r.get_last(1).size(), 0);
}

TEST(RollingLog, SingleElement) {
  R<5> r;
  r.record(10);
  auto log = r.get_last(1);
  ASSERT_EQ(log.size(), 1);
  EXPECT_EQ(log[0], 10);
}

TEST(RollingLog, SingleElementMultiAccess) {
  R<5> r;
  r.record(10);

  auto log = r.get_last(5);
  ASSERT_EQ(log.size(), 1);
  EXPECT_EQ(log[0], 10);
}

TEST(RollingLog, MultiElementMultiAccess) {
  R<5> r;
  r.record(10);
  r.record(11);
  r.record(8);

  auto log = r.get_last(5);
  ASSERT_EQ(log.size(), 3);
  EXPECT_EQ(log[0], 8);
  EXPECT_EQ(log[1], 11);
  EXPECT_EQ(log[2], 10);
}

TEST(RollingLog, MultiElementTemplateAccess) {
  R<3> r;
  r.record(1);
  r.record(2);
  r.record(3);
  r.record(4);
  r.record(5);

  auto log = r.get_last<3>();
  EXPECT_EQ(log[0], 5);
  EXPECT_EQ(log[1], 4);
  EXPECT_EQ(log[2], 3);
}

TEST_MAIN();
