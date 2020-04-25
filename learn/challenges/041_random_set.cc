// Write a data structure that provides O(1) insertion and removal, and O(1)
// access to a random element of the set with uniform probability.

#include "labm8/cpp/test.h"

#include <cstdlib>
#include <unordered_set>
#include <vector>

// Store two copies of every element, one in a hash set for O(1) insertion and
// removal, and once in a vector for O(1) random access. Insertion creates a
// new entry in both the set and the vector. Removal erases only the hash set
// entry, the vector keeps a "dead" copy. When returning a random element from
// the vector, check that it is not dead. If the ratio of dead elements in the
// vector becomes large enough, re-initialize the vector in O(n) by copying from
// the set.
template <typename T>
class RandomSet {
 private:
  std::unordered_set<T> set_;
  std::vector<T> vector_;
  size_t del_;

  void Shrink() {
    vector_.resize(set_.size());
    size_t i = 0;
    for (auto val : set_) {
      vector_[i] = val;
      ++i;
    }
  }

 public:
  RandomSet() : del_(0) {}

  size_t Size() const { return set_.size(); }

  bool Insert(const T& val) {
    if (set_.find(val) != set_.end()) {
      return false;
    }

    set_.insert(val);
    vector_.push_back(val);
    return true;
  }

  bool Remove(const T& val) {
    if (set_.find(val) == set_.end()) {
      return false;
    }

    set_.erase(val);
    ++del_;

    if (del_ / static_cast<float>(vector_.size()) > .5) {
      Shrink();
    }

    return true;
  }

  const T* Random() const {
    if (!set_.size()) {
      return nullptr;
    }

    size_t r = random() % vector_.size();
    auto it = set_.find(vector_[r]);
    if (it == set_.end()) {
      return Random();
    }

    return &vector_[r];
  }
};

TEST(RandomSet, DoubleInsert) {
  RandomSet<int> s;
  EXPECT_TRUE(s.Insert(5));
  EXPECT_FALSE(s.Insert(5));
}

TEST(RandomSet, RemoveFromEmpty) {
  RandomSet<int> s;
  EXPECT_FALSE(s.Remove(5));
}

TEST(RandomSet, InsertRemove) {
  RandomSet<int> s;
  EXPECT_TRUE(s.Insert(5));
  EXPECT_TRUE(s.Remove(5));
  EXPECT_TRUE(s.Insert(5));
  EXPECT_TRUE(s.Remove(5));
}

TEST(RandomSet, RandomNullptr) {
  RandomSet<int> s;
  EXPECT_EQ(s.Random(), nullptr);
}

TEST(RandomSet, RandomSingleElement) {
  RandomSet<int> s;
  EXPECT_TRUE(s.Insert(5));
  EXPECT_EQ(*s.Random(), 5);
  EXPECT_EQ(*s.Random(), 5);
  EXPECT_EQ(*s.Random(), 5);
}

TEST(RandomSet, RandomDoubleElement) {
  RandomSet<int> s;
  EXPECT_TRUE(s.Insert(5));
  EXPECT_TRUE(s.Insert(6));

  int r = *s.Random();
  EXPECT_TRUE(r == 5 || r == 6);

  r = *s.Random();
  EXPECT_TRUE(r == 5 || r == 6);

  r = *s.Random();
  EXPECT_TRUE(r == 5 || r == 6);
}

TEST(RandomSet, RandomRemovedElement) {
  RandomSet<int> s;
  EXPECT_TRUE(s.Insert(5));
  EXPECT_TRUE(s.Insert(6));
  EXPECT_TRUE(s.Remove(6));

  EXPECT_EQ(*s.Random(), 5);
  EXPECT_EQ(*s.Random(), 5);
  EXPECT_EQ(*s.Random(), 5);
  EXPECT_EQ(*s.Random(), 5);
}

TEST(RandomSet, Shrink) {
  RandomSet<int> s;
  EXPECT_TRUE(s.Insert(1));
  EXPECT_TRUE(s.Insert(2));
  EXPECT_TRUE(s.Insert(3));
  EXPECT_TRUE(s.Insert(4));
  EXPECT_TRUE(s.Insert(5));

  EXPECT_TRUE(s.Remove(5));
  EXPECT_TRUE(s.Remove(2));
  EXPECT_TRUE(s.Remove(4));
  EXPECT_TRUE(s.Remove(3));

  EXPECT_EQ(*s.Random(), 1);
  EXPECT_FALSE(s.Insert(1));
}

TEST_MAIN();
