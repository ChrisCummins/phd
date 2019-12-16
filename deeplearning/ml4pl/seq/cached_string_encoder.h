#pragma once

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

#include "labm8/cpp/string.h"

namespace ml4pl {

// An encoder from string to a sequence of vocabulary indices, where the results
// of encoding are stored in a persistent cache. This class is designed to be
// used in situations where you want to encode many frequently duplicate
// strings efficiently.
//
// There is no cache eviction strategy, the cache will grow without bounds. Each
// entry in the cache is equal to the size of the hashed string plus the full
// encoded result.
class CachedStringEncoder {
 public:
  CachedStringEncoder(const absl::flat_hash_map<string, int>& vocabulary)
      : _vocabulary(vocabulary),
        _unknown_element(vocabulary.size()),
        _tokens(),
        _cache() {
    // Insert each of the vocabulary keys into a set of tokens.
    for (auto item = vocabulary.begin(); item != vocabulary.end(); ++item) {
      _tokens.insert(item->first);
    }
  };

  // Encode the string, returning a vector of integers in the range
  // [0, vocabulary.size()], where vocabulary.size() represents an unknown
  // character. The length of the encoded array is in the range
  // [0, input.size()].
  std::vector<int> Encode(const std::string& input);

  // Determine if any of a set of strings starts with prefix.
  // This assumes that the prefix is not empty.
  bool HasPrefix(const std::string& prefix) const;

  // Retrieve the encoded value for an unknown token.
  int UnknownElement() const { return _unknown_element; }

 private:
  const absl::flat_hash_map<string, int> _vocabulary;
  const int _unknown_element;
  absl::flat_hash_set<string> _tokens;
  absl::flat_hash_map<string, std::vector<int>> _cache;
};

}  // namespace ml4pl
