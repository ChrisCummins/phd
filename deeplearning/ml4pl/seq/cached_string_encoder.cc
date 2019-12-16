#include "deeplearning/ml4pl/seq/cached_string_encoder.h"

namespace ml4pl {

std::vector<int> CachedStringEncoder::EncodeAndCache(const std::string& input) {
  // Check to see if there is a cache entry for the string, and if so, return
  // it.
  auto cache_entry = _cache.find(input);
  if (cache_entry != _cache.end()) {
    return cache_entry->second;
  }

  // Else encode the string, cache it, and return it.
  auto encoded = Encode(input);
  _cache.insert({input, encoded});
  return encoded;
}

std::vector<int> CachedStringEncoder::Encode(const std::string& input) const {
  // Encode a string by greedy substring matching. Using a sliding substring
  // over the input. At each step, we expand the substring one character to the
  // right and try to match a prefix in the vocabulary. If there is a match,
  // we advance another character. Once we no longer have an exact submatch, or
  // we have reached the end of the input, we retreat one character at a time,
  // attempting to match an exact token in the vocabulary. If retreat all the
  // way down to a single character without matching anything in the vocabulary,
  // we emit an "unknown vocabulary element" token and proceed.
  std::vector<int> encoded;

  int pos = 0;
  int len = 1;
  while (pos + len <= input.size()) {
    auto candidate_token = input.substr(pos, len);
    if (HasPrefix(candidate_token) && pos + len < input.size()) {
      // Prefix matched. Advance to the next character in the candidate token.
      ++len;
    } else {
      // We no longer have a match for the candidate token, or have reached
      // the end of the input. Backtrack, one character at a time, attempting
      // to find a multichar match in the vocabulary.
      while (len > 1) {
        auto backtrack_token = _vocabulary.find(input.substr(pos, len));
        if (backtrack_token != _vocabulary.end()) {
          encoded.push_back(backtrack_token->second);
          break;
        }
        --len;
      }
      // We are down to just a single character without finding a match. Process
      // the character and move on.
      if (len == 1) {
        auto backtrack_token = _vocabulary.find(input.substr(pos, len));
        if (backtrack_token == _vocabulary.end()) {
          encoded.push_back(_unknown_element);
        } else {
          encoded.push_back(backtrack_token->second);
        }
      }

      pos += len;
      len = 1;
    }
  }

  // We reached the end of the input but may not have emitted all tokens. If we
  // still have un-emitted tokens, that means that we must
  if (pos < static_cast<int>(input.size()) - 1) {
    auto tail_token = _vocabulary.find(input.substr(pos));
    if (tail_token == _vocabulary.end()) {
      auto encoded_tail = Encode(input.substr(pos, input.size() - pos));
      encoded.insert(encoded.end(), encoded_tail.begin(), encoded_tail.end());
      encoded.push_back(_unknown_element);
    } else {
      encoded.push_back(tail_token->second);
    }
  }

  return encoded;
}

bool CachedStringEncoder::HasPrefix(const std::string& prefix) const {
  for (auto& token : _tokens) {
    if (token.rfind(prefix, 0) == 0) {
      return true;
    }
  }
  return false;
}

}  // namespace ml4pl
