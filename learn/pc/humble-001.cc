#include <algorithm>
#include <forward_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using flags_list = std::forward_list<std::pair<std::string, std::string>>;

//
// Parse parameters to type 'T'.
//
// Arguments:
//
//   argv - arguments to pass
//   convert - unary operator to convert string -> T
//
// Returns:
//
//   hash map of <string,T> pairs.
//
template <typename T, typename UnaryOp>
auto parse_flags(flags_list argv, UnaryOp convert) {
  std::unordered_map<std::string, T> flags;

  std::for_each(argv.begin(), argv.end(), [&](const auto& it) {
    flags.emplace(it.first, convert(it.second));
  });

  return flags;
}

//
// Base type. Error!!
//
template <typename T>
auto parse_flags(flags_list argv);

//
// Parse int flags.
//
template <>
inline auto parse_flags<int>(flags_list argv) {
  return parse_flags<int>(argv, [](const auto& i) { return std::stoi(i); });
}

int main(int argc, char** argv) {
  try {
    // Sanity check num of arguments.
    if (!(argc % 2)) throw std::runtime_error("wrong num of arguments!");

    // Pair up arguments.
    flags_list arguments;
    for (auto it = argv + 1; it < argv + argc; it += 2)
      arguments.emplace_front(*it, *(it + 1));

    // Parse arguments.
    auto flags = parse_flags<int>(arguments);

    // Print stuff ...
    std::cout << "\nInvoked program using:\n\n  ";
    for (auto i = 0; i < argc; i++) std::cout << argv[i] << ' ';
    std::cout << "\n\nParsed " << flags.size() << " flags:\n\n";
    auto i{0};
    for (auto& pair : flags)
      std::cout << "  flag " << ++i << " = " << pair.first
                << ",  val = " << pair.second << std::endl;
  } catch (std::runtime_error& e) {
    std::cerr << e.what();
    return 1;
  }
}
