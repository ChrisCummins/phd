#include "./hash_map.h"

#include <iostream>

int main() {
  hashmap<int, char> m;

  m[1] = 'a';
  m[2] = 'b';
  m[9] = 'c';

  std::cout << "MAP: " << m << std::endl;

  return 0;
}
