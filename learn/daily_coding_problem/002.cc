// Given an array of integers, return a new array such that each element at
// index i of the new array is the product of all the numbers in the original
// array except the one at i.
//
// For example, if our input was [1, 2, 3, 4, 5], the expected output would be
// [120, 60, 40, 30, 24]. If our input was [3, 2, 1], the expected output would
// be [2, 3, 6].
//
// Follow-up: what if you can't use division?

#include <iostream>
#include <vector>

// Time: O(n)
// Space: O(n)
template <typename T>
std::vector<T> Solution(const std::vector<T>& a) {
  T product = 1;
  for (auto& x : a) {
    product *= x;
  }

  std::vector<T> out;
  out.reserve(a.size());

  for (size_t i = 0; i < a.size(); ++i) {
    out.push_back(product / a[i]);
  }

  return out;
}

template <typename T>
void PrintArray(const std::vector<T>& a) {
  for (auto& x : a) {
    std::cout << x << ", ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  PrintArray(Solution<int>({}));
  PrintArray(Solution<int>({1}));
  PrintArray(Solution<int>({1, 2}));
  PrintArray(Solution<int>({1, 2, 3}));
  PrintArray(Solution<int>({1, 2, 3, 4, 5}));
  return 0;
}
