#include <algorithm>
#include <functional>
#include <iostream>
#include <type_traits>
#include <vector>

template <typename T, typename Alloc>
std::ostream& operator<<(std::ostream& out, const std::vector<T, Alloc>& vec) {
  std::for_each(vec.begin(), vec.end(),
                [&out](auto& val) { out << val << ' '; });
  return out;
}

template <typename Container, typename T>
void map_multiply(Container& cont, const T& factor) {
  static_assert(std::is_same<typename Container::value_type, T>::value,
                "error");
  std::transform(cont.begin(), cont.end(), cont.begin(),
                 [&factor](const auto& a) { return a * factor; });
}

template <typename Container>
std::function<void(Container&)> get_doubler() {
  return [](auto& c) { return map_multiply(c, 2); };
}

int main() {
  std::vector<int> v{1, 2, 3, 4, 5};
  std::cout << v << std::endl;

  std::transform(v.begin(), v.end(), v.begin(),
                 [](const auto& a) { return a * 2; });
  std::cout << v << std::endl;

  map_multiply(v, 2);
  std::cout << v << std::endl;

  auto doubler = [](auto& cont) { return map_multiply(cont, 2); };

  doubler(v);
  std::cout << v << std::endl;

  auto doubler2 = get_doubler<decltype(v)>();
  doubler2(v);
  std::cout << v << std::endl;

  return 0;
}
