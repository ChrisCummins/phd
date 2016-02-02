/*
 * How would you design a stack which, in addition to push and pop,
 * also has a function min which returns the minimum element? Push,
 * pop and min should all operate in O(1) time.
 */
#include <algorithm>
#include <forward_list>
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
class MinStack1 {
 public:
  MinStack1() : _data(), _min(nullptr) {}

  explicit MinStack1(std::initializer_list<T> il) : MinStack1() {
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
    _data.push_back(std::move(v));

    // Update min.
    if ((_min && *_min > _data[size() - 1]) || !_min)
      _min = &_data[size() - 1];
  }

  T& min() const {
    if (!_min)  // or alternatively: if (!size()) {}
      throw std::out_of_range("Empty Stack.min()");

    return *_min;
  }

  size_t size() const {
    return _data.size();
  }

  bool empty() {
    return _data.empty();
  }

 private:
  std::vector<T> _data;
  T* _min;
};


//
// Second implementation, much more elegant. Use two singly linked
// lists: the first, storing the elements themselves (in reverse
// order, so that pop() and push() modify the list front), and the
// second storing pointers to minimum value elements. push(), pop()
// and min() are both O(1).
//
template<typename T>
class MinStack2 {
 public:
  MinStack2() {}

  explicit MinStack2(std::initializer_list<T> il) {
    for (auto& elem : il)
      push(elem);
  }

  void push(const T& elem) {
    _elem.push_front(elem);

    // Update min.
    if (_min.empty() || elem <= *_min.front())
      _min.push_front(&_elem.front());
  }

  void push(const T&& elem) {
    _elem.push_front(std::move(elem));

    // Update min.
    if (_min.empty() || elem <= *_min.front())
      _min.push_front(&_elem.front());
  }

  T& pop() {
    if (_elem.empty())
      throw std::out_of_range("MinStack2.pop()");

    auto& elem = _elem.front();
    _elem.pop_front();

    // Update min.
    if (elem == *_min.front())
      _min.pop_front();

    return elem;
  }

  T& min() {
    if (_min.empty())
      throw std::out_of_range("MinStack2.min()");

    return *_min.front();
  }

  bool empty() {
    return _elem.empty();
  }

 private:
  std::forward_list<T> _elem;
  std::forward_list<T*> _min;
};


TEST(MinStack1, tests) {
  MinStack1<int> a{5, 4, 3, 10, 9, -1, 8};

  ASSERT_EQ(-1, a.min());
  a.pop();
  a.pop();
  ASSERT_EQ(3, a.min());
  a.pop();
  a.pop();
  a.pop();
  ASSERT_EQ(4, a.min());
  a.pop();
  ASSERT_EQ(5, a.min());
}


TEST(MinStack2, tests) {
  MinStack2<int> a{5, 4, 3, 10, 9, -1, 8};

  ASSERT_EQ(-1, a.min());
  a.pop();
  a.pop();
  ASSERT_EQ(3, a.min());
  a.pop();
  a.pop();
  a.pop();
  ASSERT_EQ(4, a.min());
  a.pop();
  ASSERT_EQ(5, a.min());
}

static const size_t lengthMin = 8;
static const size_t lengthMax = 10 << 10;

void BM_MinStack1(benchmark::State& state) {
  const auto len = state.range_x();

  MinStack1<int> a;

  while (state.KeepRunning()) {
    for (int i = 0; i < len; i++)
      a.push(static_cast<int>(arc4random()));
    for (int i = 0; i < len; i++) {
      auto res = a.min();
      a.pop();
      benchmark::DoNotOptimize(res);
    }
  }
}
BENCHMARK(BM_MinStack1)->Range(lengthMin, lengthMax);

void BM_MinStack2(benchmark::State& state) {
  const auto len = state.range_x();

  MinStack2<int> a;

  while (state.KeepRunning()) {
    for (int i = 0; i < len; i++)
      a.push(static_cast<int>(arc4random()));
    for (int i = 0; i < len; i++) {
      auto res = a.min();
      a.pop();
      benchmark::DoNotOptimize(res);
    }
  }
}
BENCHMARK(BM_MinStack2)->Range(lengthMin, lengthMax);


int main(int argc, char **argv) {
  // Run unit tests:
  testing::InitGoogleTest(&argc, argv);
  const auto ret = RUN_ALL_TESTS();

  // Run benchmarks:
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return ret;
}
