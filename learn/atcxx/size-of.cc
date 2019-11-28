//
// Sizes of fundamental types (and a couple of bonus types too)
//
#include <iostream>

#define _print_size_of(name, type) \
  std::cout << "sizeof(" << name << ") = " << sizeof(type) << std::endl;
#define print_size_of(x) _print_size_of(#x, x)

int main() {
  print_size_of(bool);     // NOLINT
  print_size_of(char);     // NOLINT
  print_size_of(wchar_t);  // NOLINT
  std::cout << std::endl;

  print_size_of(int);        // NOLINT
  print_size_of(short);      // NOLINT
  print_size_of(long);       // NOLINT
  print_size_of(long long);  // NOLINT
  std::cout << std::endl;

  print_size_of(float);        // NOLINT
  print_size_of(double);       // NOLINT
  print_size_of(long double);  // NOLINT
  std::cout << std::endl;

  print_size_of(void*);   // NOLINT
  print_size_of(size_t);  // NOLINT

  return 0;
}
