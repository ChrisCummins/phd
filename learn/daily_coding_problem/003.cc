// Given an array of integers, find the first missing positive integer in linear
// time and constant space. In other words, find the lowest positive integer
// that does not exist in the array. The array can contain duplicates and
// negative numbers as well.
//
// For example, the input [3, 4, -1, 1] should give 2. The input [1, 2, 0]
// should give 3.
//
// You can modify the input array in-place.
#include <iostream>
#include <unordered_set>
#include <vector>

// Time: O(n)
// Space: O(n)
int Solution(const std::vector<int>& a) {
  std::unordered_set<int> set;
  for (auto x : a) {
    if (x > 0) {
      set.insert(x);
    }
  }

  int sum = 0;
  for (auto x : set) {
    sum += x;
  }

  int expectedSum = 0;
  for (int i = 1; i <= set.size() + 1; ++i) {
    expectedSum += i;
  }

  return expectedSum - sum;
}

int main(int argc, char** argv) {
  std::cout << Solution({0}) << std::endl;
  std::cout << Solution({0, 1, 3}) << std::endl;
  std::cout << Solution({0, 1}) << std::endl;
  std::cout << Solution({0, 1, 2}) << std::endl;
  std::cout << Solution({0, 1, 2, 2, 4}) << std::endl;
  std::cout << Solution({3, 4, -1, 1}) << std::endl;
  return 0;
}
