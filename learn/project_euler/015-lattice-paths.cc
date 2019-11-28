#include <iostream>
#include <queue>

unsigned long long routes_through_grid(const int start_x, const int start_y,
                                       const int end_x, const int end_y) {
  unsigned long long count = 0;

  if (start_x == end_x and start_y == end_y)
    ++count;
  else {
    if (start_x + 1 <= end_x)
      count += routes_through_grid(start_x + 1, start_y, end_x, end_y);
    if (start_y + 1 <= end_y)
      count += routes_through_grid(start_x, start_y + 1, end_x, end_y);
  }

  return count;
}

int main() {
  std::cout << routes_through_grid(0, 0, 1, 1) << std::endl;
  std::cout << routes_through_grid(0, 0, 2, 2) << std::endl;
  std::cout << routes_through_grid(0, 0, 3, 3) << std::endl;
  std::cout << routes_through_grid(0, 0, 20, 20) << std::endl;
}
