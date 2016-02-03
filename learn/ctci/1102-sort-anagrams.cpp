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

  for (auto &c : s) {
    if (freq.find(c) == freq.end())
      freq.emplace(c, 1);
    else
      freq[c]++;
  }

  return freq;
}

void sort_anagrams(std::vector<std::string>& arr) {
  std::vector<std::pair<std::string, freq_map>> vals;

  for (auto str : arr)
    vals.push_back(std::pair<std::string, freq_map>(
      str, get_freqcounts(str)));

  auto pred = [](const std::pair<std::string, freq_map>& a,
                 const std::pair<std::string, freq_map>& b) -> bool{
    return a.second != b.second;
  };

  std::sort(vals.begin(), vals.end(), pred);

  for (size_t i = 0; i < vals.size(); i++)
    arr[i] = vals[i].first;
}


bool is_anagram(std::string left, std::string right) {
  return get_freqcounts(left) == get_freqcounts(right);
}

TEST(anagrams, is_anagram) {
  ASSERT_TRUE(is_anagram("abc", "cab"));
  ASSERT_TRUE(is_anagram("abc", "abc"));
  ASSERT_FALSE(is_anagram("abc", "c"));
  ASSERT_FALSE(is_anagram("abc", "caba"));
}

TEST(anagrams, basic) {
  std::vector<std::string> a{"abc", "ed", "f", "de", "cab", "f"};
  sort_anagrams(a);

  ASSERT_TRUE(is_anagram(a[0], a[1]));
  ASSERT_TRUE(is_anagram(a[2], a[3]));
  ASSERT_TRUE(is_anagram(a[4], a[5]));
}

CTCI_MAIN();
