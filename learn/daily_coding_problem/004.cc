// cons(a, b) constructs a pair, and car(pair) and cdr(pair) returns the first
// and last element of that pair. For example, car(cons(3, 4)) returns 3, and
// cdr(cons(3, 4)) returns 4.
//
// Given this implementation of cons:
//
// def cons(a, b):
//     def pair(f):
//         return f(a, b)
//     return pair
// Implement car and cdr.
#include <functional>
#include <iostream>
#include <tuple>

// Time: O(1)
// Space: O(1)
template <typename T, typename Pair>
std::function<T()> car(const Pair& pair) {
  return [&]() {
    std::cout << "eval car\n";
    return std::get<0>(pair());
  };
}

// Time: O(1)
// Space: O(1)
template <typename T, typename Pair>
std::function<T()> cdr(const Pair& pair) {
  return [&]() {
    std::cout << "eval cdr\n";
    return std::get<1>(pair());
  };
}

// Time: O(1)
// Space: O(1)
template <typename T>
std::function<std::tuple<T, T>()> cons(const T& a, const T& b) {
  return [&]() {
    std::cout << "eval cons\n";
    return std::make_tuple(a, b);
  };
}

int main(int argc, char** argv) {
  auto a = car<int>(cons(3, 4));
  auto b = cdr<int>(cons(3, 4));
  std::cout << a() << std::endl;
  std::cout << b() << std::endl;
  return 0;
}
