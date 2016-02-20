#include <array>
#include <iostream>
#include <vector>

template<typename Container, unsigned int n>
class _fib {
 public:
  enum { val = _fib<Container, n - 1>::val + _fib<Container, n - 2>::val };
  static void compute(Container& result) {
    result[n] = val;
    _fib<Container, n - 1>::compute(result);
  }
};

// Partial specialization of fib for 0.
template<typename Container>
class _fib<Container, 0> {
 public:
  enum { val };
  static void compute(Container& result) { result[0] = val; }
};

// Partial specialization of fib for 1.
template<typename Container>
class _fib<Container, 1> {
 public:
  enum { val = 1 };
  static void compute(Container& result) {
    result[1] = val;
    _fib<Container, 0>::compute(result);
  }
};


//
// Compilation of fibonacci sequence at compile time.
//
template<typename T, unsigned int n>
constexpr auto compile_time_fib() {
  static_assert(n > 0, "compile_time_fib(): argument < 1");

  std::array<T, n> result;
  _fib<decltype(result), n>::compute(result);
  return result;
}


int main() {
  auto seq = compile_time_fib<uint64_t, 46>();

  for (decltype(seq)::size_type i = 0; i < seq.size(); ++i)
    std::cout << "fib(" << (i + 1) << ") = " << seq[i] << std::endl;
}
