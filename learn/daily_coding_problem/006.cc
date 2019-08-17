// Given the mapping a = 1, b = 2, ... z = 26, and an encoded message, count the
// number of ways it can be decoded.
//
// For example, the message '111' would give 3, since it could be decoded as
// 'aaa', 'ka', and 'ak'.
//
// You can assume that the messages are decodable. For example, '001' is not
// allowed.
#include <functional>
#include <iostream>
#include <string>
#include <vector>

// Time: O(2 ^ n)
// Space: O(n)
int RecursiveSolutionHelper(const std::string& s, const int& n) {
  if (n == 0 || n == 1) {
    return 1;
  }

  int count = 0;
  if (s[n - 1] != '0') {
    count += RecursiveSolutionHelper(s, n - 1);
  }

  if (s[n - 2] == '1' || (s[n - 2] == '2' && s[n - 1] <= '6')) {
    count += RecursiveSolutionHelper(s, n - 2);
  }

  return count;
}

int RecursiveSolution(const std::string& s) {
  return RecursiveSolutionHelper(s, s.size());
}

// The idea is to move from front to back, using dynamic programming to
// cache previous results. However, since we only re-use the last n-2 elements,
// we can replace the DP array with a pair of variables and rotate them.
//
// Time: O(n)
// Space: O(1)
int IterativeSolution(const std::string& s) {
  std::vector<int> counts(s.size() + 1, 0);

  int a = 1, b = 1;  // counts[0] and counts[1]
  for (int i = 2; i <= s.size(); ++i) {
    int count = 0;

    if (s[i - 1] != '0') {
      count += b;
    }

    if (s[i - 2] == '1' || (s[i - 2] == '2' && s[i - 1] <= '6')) {
      count += a;
    }

    // Rotate variables.
    a = b;
    b = count;
  }

  return b;
}

void Solve(std::function<int(const std::string& s)> fn, const std::string& a) {
  auto s = fn(a);
  for (auto& x : a) {
    std::cout << x << ", ";
  }
  std::cout << " = " << s << std::endl;
}

int main(int argc, char** argv) {
  std::cout << "Recursive:\n";
  Solve(RecursiveSolution, "");
  Solve(RecursiveSolution, "1");
  Solve(RecursiveSolution, "26");
  Solve(RecursiveSolution, "111");
  Solve(RecursiveSolution, "121");
  Solve(RecursiveSolution, "1111");

  std::cout << "Dynamic programming:\n";
  Solve(IterativeSolution, "");
  Solve(IterativeSolution, "1");
  Solve(IterativeSolution, "26");
  Solve(IterativeSolution, "111");
  Solve(IterativeSolution, "121");
  Solve(IterativeSolution, "1111");
  return 0;
}
