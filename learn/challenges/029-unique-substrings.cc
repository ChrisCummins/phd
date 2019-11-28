// Given a list of strings, build a list of       NOLINT(legal/copyright)
// minimal substrings, required to uniquely disambiguate each string.
// Each substring must begin with the first character of its respective string.

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

std::unordered_map<std::string, int> histogram(
    const std::vector<std::string>& input) {
  std::unordered_map<std::string, int> m;

  for (const auto& s : input) {
    if (m.find(s) == m.end()) {
      m.insert({s, 1});
    } else {
      m[s] += 1;
    }
  }

  return m;
}

void _substrings(const std::vector<std::string>& input,
                 std::vector<std::string>* substrings) {
  while (1) {
    bool repeat = false;
    auto h = histogram(*substrings);

    for (size_t i = 0; i < input.size(); ++i) {
      auto freq = h[(*substrings)[i]];

      if (freq > 1) {
        if (!input[i].size() || ((*substrings)[i].size() == input[i].size())) {
          // if substring is already the length of the entire string, empty the
          // list of substrings and return.
          (*substrings).clear();
          return;
        } else {
          auto next_char = input[i][(*substrings)[i].size()];
          (*substrings)[i].push_back(next_char);
          repeat = true;
        }
      }
    }

    if (!repeat) return;
  }
}

std::vector<std::string> substrings(const std::vector<std::string>& input) {
  // If strings cannot be uniquely disambiguated, raises an error.
  std::vector<std::string> out;

  for (auto& s : input) {
    out.push_back(std::string(1, s[0]));
  }

  _substrings(input, &out);

  return out;
}

TEST(unique_substrings, empty) {
  const std::vector<std::string> in;

  auto out = substrings(in);
  ASSERT_EQ(out.size(), 0);
}

TEST(unique_substrings, one_char) {
  const std::vector<std::string> in{"a"};

  auto out = substrings(in);
  ASSERT_EQ(out.size(), 1);
  ASSERT_EQ(out[0], "a");
}

TEST(unique_substrings, one) {
  const std::vector<std::string> in{"abc"};

  auto out = substrings(in);
  ASSERT_EQ(out.size(), 1);
  ASSERT_EQ(out[0], "a");
}

TEST(unique_substrings, two) {
  const std::vector<std::string> in{"a", "b"};

  auto out = substrings(in);
  ASSERT_EQ(out.size(), 2);
  ASSERT_EQ(out[0], "a");
  ASSERT_EQ(out[1], "b");
}

TEST(unique_substrings, two_char) {
  const std::vector<std::string> in{"ab", "ac"};

  auto out = substrings(in);
  ASSERT_EQ(out.size(), 2);
  ASSERT_EQ(out[0], "ab");
  ASSERT_EQ(out[1], "ac");
}

TEST(unique_substrings, three_char) {
  const std::vector<std::string> in{"aab", "acc"};

  auto out = substrings(in);
  ASSERT_EQ(out.size(), 2);
  ASSERT_EQ(out[0], "aa");
  ASSERT_EQ(out[1], "ac");
}

TEST(unique_substrings, four_char) {
  const std::vector<std::string> in{"a", "ba", "bbe", "bbd", "c", "be"};

  auto out = substrings(in);
  ASSERT_EQ(out.size(), 6);
  ASSERT_EQ(out[0], "a");
  ASSERT_EQ(out[1], "ba");
  ASSERT_EQ(out[2], "bbe");
  ASSERT_EQ(out[3], "bbd");
  ASSERT_EQ(out[4], "c");
  ASSERT_EQ(out[5], "be");
}

TEST(unique_substrings, nonunique) {
  const std::vector<std::string> in{"a", "a"};

  auto out = substrings(in);
  ASSERT_EQ(out.size(), 0);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
