// first triangle number to over have 500 divisors
#include <iostream>
#include <limits>
#include <vector>

using num_t = int;

int num_divisors(const num_t n) {
  // T(n) = O(n)
  // S(n) = O(1)
  int divisors = 1;

  for (int i = 2; i <= n; ++i) {
    if (!(n % i)) {
      ++divisors;
    }
  }

  return divisors;
}

num_t triangle_number(const int min_divisors) {
  // simple, brute force approach
  //
  // T(n) = O(n**2)
  // S(n) = O(1)
  num_t triangle = 0;

  for (num_t i = 1; i < std::numeric_limits<num_t>::max(); ++i) {
    triangle += i;
    if (num_divisors(triangle) > min_divisors) return triangle;
  }
  throw std::invalid_argument("reached value max");
}

int main(int argc, char **argv) {
  std::cout << triangle_number(500) << std::endl;
}
