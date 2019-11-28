#include <array>
#include <iostream>
#include <string>
#include <vector>

template <typename Container, unsigned int n>
class _fib {
 public:
  enum { val = _fib<Container, n - 1>::val + _fib<Container, n - 2>::val };
  static void compute(Container& result) {
    result[n] = val;
    _fib<Container, n - 1>::compute(result);
  }
};

// Partial specialization of fib for 0.
template <typename Container>
class _fib<Container, 0> {
 public:
  enum { val };
  static void compute(Container& result) { result[0] = val; }
};

// Partial specialization of fib for 1.
template <typename Container>
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
template <typename T, unsigned int n>
constexpr auto compile_time_fib() {
  static_assert(n > 0, "compile_time_fib(): argument < 1");

  auto result = std::array<T, n>();
  _fib<decltype(result), n>::compute(result);
  return result;
}

//
// A contrived "template template" parameters example.
//
template <template <typename, typename> class vector, typename T, typename A>
void print_vec(const vector<T, A>& vec) {
  for (typename vector<T, A>::size_type i = 0; i < vec.size(); ++i) {
    const T& val = vec[i];
    std::cout << val << ' ';
  }
  std::cout << std::endl;
}

//
// A contrived "enable_if" example. Typename T must be "int" type.
//
template <typename T,
          typename = typename std::enable_if<std::is_same<T, int>::value>::type>
void _print_int(const std::string name, const T& val) {
  std::cout << name << " = " << val << std::endl;
}

#define print_int(x) _print_int((#x), (x));

int main() {
  auto seq = compile_time_fib<uint64_t, 46>();

  for (decltype(seq)::size_type i = 0; i < seq.size(); ++i)
    std::cout << "fib(" << (i + 1) << ") = " << seq[i] << std::endl;

  std::vector<int> v{1, 2, 3, 4, 5};
  print_vec(v);

  auto a = 5, b = 3;
  print_int(a);
  print_int(b);
  print_int(a + b);
}
