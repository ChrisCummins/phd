// is_integral:

template <typename T>
struct is_integral {
  static const bool value = false;
};

template <>
struct is_integral<int> {
  static const bool value = true;
};

// enable_if:

template <bool B, typename T = void>
struct enable_if {};

template <typename T>
struct enable_if<true, T> {
  using type = T;
};

// test:

template <typename T,
          typename enable_if<is_integral<T>::value>::type* = nullptr>
int foobar(const T& val) {
  return 0;
}

int main() {
  static_assert(is_integral<int>::value, "error");
  static_assert(is_integral<float>::value == false, "error");

  foobar(3);
  foobar(-1);

  // Return statement is optional in C++
}
