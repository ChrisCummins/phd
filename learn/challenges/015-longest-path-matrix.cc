/* Find the longest path in a matrix with given constraints
 *
 * http://www.geeksforgeeks.org/find-the-longest-path-in-a-matrix-with-given-constraints/
 */
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

template <size_t nrows, size_t ncols>
void longest_ascending_route(int (&mat)[nrows][ncols], int j, int i,
                             std::vector<int>& stack) {
  stack.push_back(mat[j][i]);

#define _is_in_bounds(y, x) \
  ((j + (y)) >= 0 && (j + (y)) < nrows && (i + (x)) >= 0 && (i + (x)) < ncols)
#define _satisfies_constraint(y, x) (mat[j + (y)][i + (x)] == mat[j][i] + 1)
#define _is_valid(y, x) (_is_in_bounds(y, x) && _satisfies_constraint(y, x))

  if (_is_valid(-1, 0)) longest_ascending_route(mat, j - 1, i, stack);  // go up
  if (_is_valid(1, 0))
    longest_ascending_route(mat, j + 1, i, stack);  // go down
  if (_is_valid(0, -1))
    longest_ascending_route(mat, j, i - 1, stack);  // go left
  if (_is_valid(0, 1))
    longest_ascending_route(mat, j, i + 1, stack);  // go right

#undef _is_in_bounds
#undef _satisfies_constraint
#undef _is_valid
}

template <typename T>
bool list_length(const T& a, const T& b) {
  return a.size() < b.size();
}

template <size_t nrows, size_t ncols>
std::vector<int> solve(int (&mat)[nrows][ncols]) {
  std::vector<std::vector<int>> solutions;

  for (int i = 0; i < nrows * ncols; ++i) {
    std::vector<int> solution;
    longest_ascending_route(mat, i / ncols, i % ncols, solution);
    solutions.push_back(solution);
  }

  return *std::max_element(solutions.begin(), solutions.end(),
                           list_length<std::vector<int>>);
}

int main(int argc, char** argv) {
  const size_t ncols = 3, nrows = 3;
  int mat[ncols][nrows] = {{1, 2, 9}, {5, 3, 8}, {4, 6, 7}};

  std::cout << "Input:  mat[][] = {";
  for (size_t j = 0; j < nrows; ++j) {
    std::cout << "{";
    for (size_t i = 0; i < ncols; ++i) {
      std::cout << mat[j][i];
      if (i != ncols - 1) std::cout << ", ";
    }
    if (j != nrows - 1) std::cout << "},\n                   ";
  }
  std::cout << "}}" << std::endl;

  std::vector<int> solution = solve(mat);

  std::cout << "Output: " << solution.size() << std::endl
            << "The longest path is ";
  for (size_t i = 0; i < solution.size(); ++i) {
    std::cout << solution[i];
    if (i != solution.size() - 1) std::cout << '-';
  }
  std::cout << '.' << std::endl;

  assert(solution[0] == 6);
  assert(solution[1] == 7);
  assert(solution[2] == 8);
  assert(solution[3] == 9);

  return 0;
}
