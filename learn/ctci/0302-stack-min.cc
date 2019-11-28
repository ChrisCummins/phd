/*
 * How would you design a stack which, in addition to push and pop,
 * also has a function min which returns the minimum element? Push,
 * pop and min should all operate in O(1) time.
 */
#include "./ctci.h"

#include <algorithm>
#include <forward_list>
#include <vector>

static unsigned int seed = 0xCEC;

//
// Use two singly linked lists: the first, storing the elements
// themselves (in reverse order, so that pop() and push() modify the
// list front), and the second storing pointers to minimum value
// elements. push(), pop() and min() are all O(1).
//
template <typename T>
class MinStack {
 public:
  MinStack() {}

  explicit MinStack(std::initializer_list<T> il) {
    for (auto& elem : il) push(elem);
  }

  void push(const T& elem) {
    _elem.push_front(elem);

    // Update min.
    if (_min.empty() || elem <= *_min.front()) _min.push_front(&_elem.front());
  }

  void push(const T&& elem) {
    _elem.push_front(std::move(elem));

    // Update min.
    if (_min.empty() || elem <= *_min.front()) _min.push_front(&_elem.front());
  }

  const T& pop() {
    if (_elem.empty()) throw std::out_of_range("MinStack.pop()");

    auto& elem = _elem.front();
    _elem.pop_front();

    // Update min.
    if (elem == *_min.front()) _min.pop_front();

    return elem;
  }

  const T& min() {
    if (_min.empty()) throw std::out_of_range("MinStack.min()");

    return *_min.front();
  }

  bool empty() { return _elem.empty(); }

 private:
  std::forward_list<T> _elem;
  std::forward_list<T*> _min;
};

//
// My first attempt at a solution. Store a pointer to the min
// element. This is cheap for inserts, but requires pop() to search
// for a new min if the popped element is the minimum. This also uses
// a vector which is expensive for repeated inserts.
//
template <typename T>
class MinStack_alt {
 public:
  MinStack_alt() : _data(), _min(nullptr) {}

  explicit MinStack_alt(std::initializer_list<T> il) : MinStack_alt() {
    for (auto& v : il) push(v);
  }

  const T& pop() {
    if (!size()) throw std::out_of_range("Empty Stack.pop()");

    auto v = &_data[size() - 1];
    _data.pop_back();

    // Update min.
    if (v == _min) _min = &(*std::min_element(_data.begin(), _data.end()));

    return *v;
  }

  void push(const T& v) {
    _data.push_back(v);

    // Update min.
    if ((_min && *_min > v) || !_min) _min = &_data[size() - 1];
  }

  void push(const T&& v) {
    _data.push_back(std::move(v));

    // Update min.
    if ((_min && *_min > _data[size() - 1]) || !_min) _min = &_data[size() - 1];
  }

  const T& min() const {
    if (!_min)  // or alternatively: if (!size()) {}
      throw std::out_of_range("Empty Stack.min()");

    return *_min;
  }

  size_t size() const { return _data.size(); }

  bool empty() { return _data.empty(); }

 private:
  std::vector<T> _data;
  T* _min;
};

///////////
// Tests //
///////////

TEST(MinStack, tests) {
  MinStack<int> a{5, 4, 3, 10, 9, -1, 8};

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

TEST(MinStack_alt, tests) {
  MinStack_alt<int> a{5, 4, 3, 10, 9, -1, 8};

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

////////////////
// Benchmarks //
////////////////

static const size_t BM_length_min = 8;
static const size_t BM_length_max = 5 << 10;

void BM_MinStack(benchmark::State& state) {
  const auto len = state.range(0);
  MinStack<int> a;

  while (state.KeepRunning()) {
    for (int i = 0; i < len; i++) a.push(static_cast<int>(rand_r(&seed)));
    for (int i = 0; i < len; i++) {
      auto res = a.min();
      a.pop();
      benchmark::DoNotOptimize(res);
    }
  }
}
BENCHMARK(BM_MinStack)->Range(BM_length_min, BM_length_max);

void BM_MinStack_alt(benchmark::State& state) {
  const auto len = state.range(0);
  MinStack_alt<int> a;

  while (state.KeepRunning()) {
    for (int i = 0; i < len; i++) a.push(static_cast<int>(rand_r(&seed)));
    for (int i = 0; i < len; i++) {
      auto res = a.min();
      a.pop();
      benchmark::DoNotOptimize(res);
    }
  }
}
BENCHMARK(BM_MinStack_alt)->Range(BM_length_min, BM_length_max);

CTCI_MAIN();
