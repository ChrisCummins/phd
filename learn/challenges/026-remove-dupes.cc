#include <iostream>
#include <set>
#include <vector>

//
// Given an ordered vector, return a vector with duplicates removed.
//
// T(n) = O(in)
// S(n) = O(in)
//
template <typename T>
std::vector<T> approach1(const std::vector<T>& in) {
  std::vector<T> out;
  const T* last;

  for (auto& v : in) {
    if (last) {
      if (*last != v) {
        out.push_back(v);
      }
    } else {
      out.push_back(v);
    }

    last = &v;
  }

  return out;
}

//
// Given an *unordered* vector, return an ordered vector with duplicates
// removed.
//
// T(n) = O(n), O(n*log(n)) worst case
// S(n) = O(n)
//
template <typename T>
std::vector<T> approach2(const std::vector<T>& in) {
  std::vector<T> out;
  std::set<T> set;

  for (auto& v : in) set.insert(v);

  for (auto& v : set) out.push_back(v);

  return out;
}

//
// Given an ordered vector, remove duplicates in-place.
//
// T(n) = O(n)
// S(n) = O(1)
//
template <typename T>
void approach3(std::vector<T>& in) {
  size_t slow = 0, fast = 1;

  if (!in.size()) return;

  for (; fast < in.size(); fast++) {
    if (in[fast] != in[slow]) in[++slow] = in[fast];
  }

  in[++slow] = in[--fast];
  in.resize(slow);
}

void print_vec(const std::vector<int>& V) {
  for (auto& v : V) std::cout << v << ' ';
  std::cout << std::endl;
}

int main() {
  std::vector<int> a = {1, 2, 3, 4, 5};
  std::vector<int> a2 = {1, 1, 2, 3, 4, 4, 4, 4, 5, 5};

  print_vec(a);
  print_vec(approach1(a2));
  print_vec(approach2(a2));

  approach3(a2);
  print_vec(a2);
}
