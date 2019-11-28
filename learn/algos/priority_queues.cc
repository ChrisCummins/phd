#include <iostream>
#include <vector>

#include <gtest/gtest.h>

// Unsorted array backing
template <typename T>
class priority_queue_unsorted_array {
 public:
  void insert(const T& val) {
    _data.push_back(val);

    if (_min_idx < 0) {
      _min_idx = 0;
    } else if (_data.size() > 1 && val < min()) {
      _min_idx = _data.size() - 1;
    }
  }

  priority_queue_unsorted_array() : _data(), _min_idx(-1) {}

  const T& min() const { return _data[_min_idx]; }

  void delete_min() {
    _data.erase(_data.begin() + _min_idx);

    // set new min_idx
    if (!_data.size()) {
      _min_idx = -1;
    } else {
      int m = 0;
      for (int i = 1; i < _data.size(); ++i)
        if (_data[i] < _data[m]) m = i;
      _min_idx = m;
    }
  }

 private:
  std::vector<T> _data;
  int _min_idx;
};

TEST(priority_queue, unsorted_array) {
  priority_queue_unsorted_array<int> p;

  p.insert(10);
  ASSERT_EQ(p.min(), 10);
  p.insert(9);
  ASSERT_EQ(p.min(), 9);
  p.insert(8);
  ASSERT_EQ(p.min(), 8);
  p.insert(1);
  ASSERT_EQ(p.min(), 1);
  p.insert(9);
  ASSERT_EQ(p.min(), 1);
  p.insert(10);
  ASSERT_EQ(p.min(), 1);
  p.insert(8);
  ASSERT_EQ(p.min(), 1);
  p.delete_min();
  ASSERT_EQ(p.min(), 8);
  p.delete_min();
  ASSERT_EQ(p.min(), 8);
  p.delete_min();
  ASSERT_EQ(p.min(), 9);
  p.delete_min();
  ASSERT_EQ(p.min(), 9);
  p.delete_min();
  ASSERT_EQ(p.min(), 10);
}

// Sorted array backing
template <typename T>
class priority_queue_sorted_array {
 public:
  void insert(const T& val) {
    if (!_data.size()) {
      _data.push_back(val);
    } else {
      for (int i = 0; i < _data.size(); ++i) {
        if (_data[i] > val) {
          _data.insert(_data.begin() + i, val);
          break;
        }
      }
    }
  }

  const T& min() const { return _data[0]; }

  void delete_min() { _data.erase(_data.begin()); }

 private:
  std::vector<T> _data;
};

TEST(priority_queue, sorted_array) {
  priority_queue_sorted_array<int> p;

  p.insert(10);
  ASSERT_EQ(p.min(), 10);
  p.insert(9);
  ASSERT_EQ(p.min(), 9);
  p.insert(8);
  ASSERT_EQ(p.min(), 8);
  p.insert(1);
  ASSERT_EQ(p.min(), 1);
  p.insert(9);
  ASSERT_EQ(p.min(), 1);
  p.insert(10);
  ASSERT_EQ(p.min(), 1);
  p.insert(8);
  ASSERT_EQ(p.min(), 1);
  p.delete_min();
  ASSERT_EQ(p.min(), 8);
  p.delete_min();
  ASSERT_EQ(p.min(), 8);
  p.delete_min();
  ASSERT_EQ(p.min(), 9);
  p.delete_min();
  ASSERT_EQ(p.min(), 9);
  p.delete_min();
  ASSERT_EQ(p.min(), 10);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
