// Given an array of n integers, generate an int not in the input.
#include <vector>
#include <iostream>
#include <algorithm>

// Solution 1: For i in 1..inf, scan through array to see if i
// in array. If not, return it; if yes, repeat.
// Time: O(n * n)
// Space: O(1)
// Solution 2: Sort the array, return a[0] - 1.
// Time: O(n log n)
// Space: O(n)

// Solution 3: Find the max element in the set, return max + 1.
// Time: O(n)
// Space: O(1)
int Solution(const std::vector<int>& a) {
  auto max = std::max_element(a.begin(), a.end());
  if (max == a.end()) {
    return 0;
  } else {
    return *max + 1;
  }
}


int main(int argc, char** argv) {
  std::cout << Solution({}) << std::endl;
  std::cout << Solution({1}) << std::endl;
  std::cout << Solution({1, 2}) << std::endl;
  std::cout << Solution({-1, 0, 10}) << std::endl;
  return 0;
}
