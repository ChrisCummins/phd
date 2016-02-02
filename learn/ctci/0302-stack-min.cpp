/*
 * How would you design a stack which, in addition to push and pop,
 * also has a function min which returns the minimum element? Push,
 * pop and min should all operate in O(1) time.
 */
#include <algorithm>
#include <list>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop


//
// First implementation. Store a pointer to the min element. This is
// cheap for inserts, but requires pop() to search for a new min if
// the popped element is the minimum.
//
template<typename T>
class Stack1 {
 public:
  Stack1() : _data(), _min(nullptr) {}

  explicit Stack1(std::initializer_list<T> il) : Stack1() {
    for (auto& v : il)
      push(v);
  }

  T& pop() {
    if (!size())
      throw std::out_of_range("Empty Stack.pop()");

    auto v = &_data[size() - 1];
    _data.pop_back();

    // Update min.
    if (v == _min)
      _min = &(*std::min_element(_data.begin(), _data.end()));

    return *v;
  }

  void push(const T& v) {
    _data.push_back(v);

    // Update min.
    if ((_min && *_min > v) || !_min)
      _min = &_data[size() - 1];
  }

  void push(const T&& v) {
    push(std::move(v));
  }

  T& min() const {
    if (!_min)  // or alternatively: if (!size()) {}
      throw std::out_of_range("Empty Stack.min()");

    return *_min;
  }

  size_t size() const {
    return _data.size();
  }

 private:
  std::vector<T> _data;
  T* _min;
};

TEST(challenge, Stack1) {
  Stack1<int> s{1, 2, 3, 4, 5, -10, 6};

  ASSERT_EQ(-10, s.min());

  s.pop();
  s.pop();

  ASSERT_EQ(1, s.min());
}

int main(int argc, char **argv) {
  // Run unit tests:
  testing::InitGoogleTest(&argc, argv);
  const auto ret = RUN_ALL_TESTS();

  // Run benchmarks:
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return ret;
}
