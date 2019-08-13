// Given a list of numbers and a number k, return whether any two numbers from
// the list add up to k.
//
// For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is
// 17.
//
// Bonus: Can you do this in one pass?
#include <algorithm>
#include <iostream>
#include <vector>

// Time: O(n log n)
// Space: O(n)
template <typename T>
bool SumsToK(const std::vector<T>& input, const T& k) {
  int i = 0, j = input.size() - 1;

  std::vector<T> a(input.begin(), input.end());
  std::sort(a.begin(), a.end());

  while (i < j) {
    const T s = a[i] + a[j];
    if (s == k) {
      return true;
    } else if (s < k) {
      ++i;
    } else if (s > k) {
      --j;
    }
  }

  return false;
}

template <typename T>
void Test(const std::vector<T>& a, const T& k, const bool& expected) {
  auto actual = SumsToK(a, k);
  if (actual == expected) {
    std::cout << "Test: passed" << std::endl;
  } else {
    std::cout << "Test: failed. Expected = " << expected
              << ", actual = " << actual << std::endl;
  }
}

int main(int argc, char** argv) {
  Test({}, 0, false);
  Test({1, 2, 3}, 1, false);
  Test({1, 2, 3}, 4, true);
  Test({1, 2, 3}, 5, true);
  Test({10, 15, 3, 7}, 17, true);
  Test({10, 15, 3, 7}, 12, false);
  return 0;
}
