#include "deeplearning/ml4pl/seq/cached_string_encoder.h"

#include "labm8/cpp/logging.h"

namespace ml4pl {

std::vector<int> CachedStringEncoder::Encode(const std::string& input) {
  // Check to see if there is a cache entry for the string, and if so, return
  // it.
  auto cache_entry = _cache.find(input);
  if (cache_entry != _cache.end()) {
    return cache_entry->second;
  }

  std::vector<int> encoded;

  int pos = 0;
  int len = 2;
  while (pos < input.size()) {
    auto candidate_token = input.substr(pos, len);
    if (HasPrefix(candidate_token)) {
      // Prefix matched.

      if (candidate_token.size() < len) {
        // We have reached the end of the string. Emit an encoded token or
        // unknown elements.
        auto token = _vocabulary.find(candidate_token);
        if (token == _vocabulary.end()) {
          for (int i = 0; i < candidate_token.size(); ++i) {
            encoded.push_back(_unknown_element);
          }
        } else {
          encoded.push_back(token->second);
        }
        break;
      }

      // Advance to the next character in the candidate token.
      ++len;
    } else {
      // Candidate token doesn't match any prefixes in the vocabulary, so either
      // emit the last known good token, or unknown tokens.
      auto backtrack_token = _vocabulary.find(input.substr(pos, len - 1));
      if (backtrack_token == _vocabulary.end()) {
        for (int i = 0; i < len - 1; ++i) {
          encoded.push_back(_unknown_element);
        }
      } else {
        encoded.push_back(backtrack_token->second);
      }

      // Advance.
      pos = pos + len - 1;
      len = 2;
    }
  }

  // Insert the encoded text into the cache.
  _cache.insert({input, encoded});

  return encoded;
}

bool CachedStringEncoder::HasPrefix(const std::string& prefix) {
  for (auto& token : _tokens) {
    if (token.rfind(prefix, 0) == 0) {
      return true;
    }
  }
  return false;
}

}  // namespace ml4pl
