/*
 * Write a method to sort an array of strings so that all the anagrams
 * are next to each other.
 */
#include "./ctci.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using freq_map = std::unordered_map<char, size_t>;

freq_map get_freqcounts(std::string s) {
  freq_map freq;

  for (auto& c : s) {
    if (freq.find(c) == freq.end())
      freq.emplace(c, 1);
    else
      freq[c]++;
  }

  return freq;
}

//
// First solution. Map strings to frequency counts, then sort these
// mapped values so that equal frequency counts are contiguous.
//
// O(n log (n)) time, O(n) space.
//
void sort_anagrams(std::vector<std::string>& arr) {
  using pair_type = std::pair<std::string, freq_map>;
  std::vector<pair_type> vals;

  for (auto str : arr) vals.push_back(pair_type(str, get_freqcounts(str)));

  std::sort(vals.begin(), vals.end(),
            [](const pair_type& a, const pair_type& b) {
              return a.second != b.second;
            });

  for (size_t i = 0; i < vals.size(); i++) arr[i] = vals[i].first;
}

bool is_anagram(std::string left, std::string right) {
  return get_freqcounts(left) == get_freqcounts(right);
}

///////////
// Tests //
///////////

TEST(anagrams, is_anagram) {
  ASSERT_TRUE(is_anagram("abc", "cab"));
  ASSERT_TRUE(is_anagram("abc", "abc"));
  ASSERT_FALSE(is_anagram("abc", "c"));
  ASSERT_FALSE(is_anagram("abc", "caba"));
}

CTCI_MAIN();
