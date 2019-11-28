#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "learn/absl/message.pb.h"

#include <cstdlib>
#include <iostream>
#include <string>

using std::string;

#define FATAL(...)                           \
  std::cerr << absl::StrFormat(__VA_ARGS__); \
  exit(1);

template <typename Container, typename K>
auto FindInMap(const Container& container, const K& key) {
  auto it = container.find(key);
  if (it == container.end()) {
    FATAL("Key not found");
  }
  return (*it).second;
}

int main() {
  absl::flat_hash_map<string, int> map;

  map.insert({"Answer", 42});

  std::cout << absl::StrFormat("Hello, %s!\n", "world");

  auto value = FindInMap(map, "Answer");

  std::cout << absl::StrFormat("The answer is %d\n", value);

  MyMessage proto;
  proto.set_name("foo");
  std::cout << "Proto: " << proto.DebugString();

  return 0;
}
