#include <stdio.h>

int is_prime(const int val) {
  // T(n) = O(n)
  // S(n) = O(1)
  //
  // return 1 if prime, else 0
  for (int i = 2; i < val; ++i) {
    if (!(val % i)) {
      return 0;
    }
  }

  return 1;
}

unsigned long long sum_of_primes_below(const int max) {
  // T(n) = O(n**2)
  // S(n) = O(1)
  unsigned long long sum = 0;

  for (int i = 2; i < max; ++i) {
    if (is_prime(i)) {
      sum += i;
    }
  }

  return sum;
}

int main(int argc, char **argv) {
  int max = 2000000;
  unsigned long long sum = sum_of_primes_below(max);

  printf("sum of primes below %d is %llu\n", max, sum);
}
