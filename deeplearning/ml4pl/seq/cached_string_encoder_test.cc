#include "deeplearning/ml4pl/seq/cached_string_encoder.h"

#include <vector>

#include "absl/container/flat_hash_map.h"

#include "labm8/cpp/string.h"
#include "labm8/cpp/test.h"

namespace ml4pl {
namespace {

TEST(CachedStringEncoder, EncodeEmptyString) {
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"a", 0});

  CachedStringEncoder encoder(vocabulary);

  auto encoded_1 = encoder.EncodeAndCache("");
  ASSERT_EQ(0, encoded_1.size());
}

TEST(CachedStringEncoder, EncodeUnknownElements) {
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"a", 0});

  CachedStringEncoder encoder(vocabulary);

  auto encoded_1 = encoder.EncodeAndCache("bc");
  ASSERT_EQ(2, encoded_1.size());
  ASSERT_EQ(encoder.UnknownElement(), encoded_1[0]);
  ASSERT_EQ(encoder.UnknownElement(), encoded_1[1]);
}

TEST(CachedStringEncoder, EncodeMultiCharMatch) {
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"int", 0});

  CachedStringEncoder encoder(vocabulary);

  auto encoded = encoder.EncodeAndCache("int");
  ASSERT_EQ(1, encoded.size());
  ASSERT_EQ(0, encoded[0]);
}

TEST(CachedStringEncoder, EncodeMultiCharMatchComposite) {
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"int", 0});
  vocabulary.insert({" ", 1});

  CachedStringEncoder encoder(vocabulary);

  auto encoded = encoder.EncodeAndCache("int int");
  ASSERT_EQ(3, encoded.size());
  ASSERT_EQ(0, encoded[0]);
  ASSERT_EQ(1, encoded[1]);
  ASSERT_EQ(0, encoded[2]);
}

TEST(CachedStringEncoder, CachedEncodeOfSingleElementString) {
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"a", 0});

  CachedStringEncoder encoder(vocabulary);

  // Test encoding an input.
  auto encoded_1 = encoder.EncodeAndCache("a");
  ASSERT_EQ(1, encoded_1.size());
  ASSERT_EQ(0, encoded_1[0]);

  // Test encoding the same input again.
  auto encoded_2 = encoder.EncodeAndCache("a");
  ASSERT_EQ(1, encoded_2.size());
  ASSERT_EQ(0, encoded_2[0]);
}

TEST(CachedStringEncoder, UnknownFinalToken) {
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"int", 0});

  CachedStringEncoder encoder(vocabulary);

  auto encoded = encoder.EncodeAndCache("int ");
  ASSERT_EQ(2, encoded.size());
  ASSERT_EQ(0, encoded[0]);
  ASSERT_EQ(encoder.UnknownElement(), encoded[1]);
}

TEST(CachedStringEncoder, UnknownFinalTokens) {
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"int", 0});

  CachedStringEncoder encoder(vocabulary);

  auto encoded = encoder.EncodeAndCache("int   ");
  ASSERT_EQ(4, encoded.size());
  ASSERT_EQ(0, encoded[0]);
  ASSERT_EQ(encoder.UnknownElement(), encoded[1]);
  ASSERT_EQ(encoder.UnknownElement(), encoded[2]);
  ASSERT_EQ(encoder.UnknownElement(), encoded[3]);
}

TEST(CachedStringEncoder, UnknownFinalTokenWithMatchedPrefix) {
  // Vocabulary: {"aaaab": 0, "aaaac": 1, "aaaad": 2, "aa": 3}
  // Input: "aaaae"
  // Expected behaviour:
  //   1. Forward prefix match "aa".
  //   2. Forward prefix match "aaa".
  //   3. Forward prefix match "aaaa".
  //   4. No prefix match "aaaae".
  //   5. No backward match "aaaa".
  //   6. No backward match "aaa".
  //   7. Emit "aa" -> 3.
  //   8. Forward prefix match "aa".
  //   9. No prefix match "aaa".
  //  10. Emit "aa" -> 3.
  //  11. No prefix match "e".
  //  12. Emit "e" -> unknown.
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"aaaab", 0});
  vocabulary.insert({"aaaac", 1});
  vocabulary.insert({"aaaad", 2});
  vocabulary.insert({"aa", 3});

  CachedStringEncoder encoder(vocabulary);

  auto encoded = encoder.EncodeAndCache("aaaae");
  ASSERT_EQ(3, encoded.size());
  ASSERT_EQ(3, encoded[0]);
  ASSERT_EQ(3, encoded[1]);
  ASSERT_EQ(encoder.UnknownElement(), encoded[2]);
}

TEST(CachedStringEncoder, GreedySubstringTail) {
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"aaa", 0});
  vocabulary.insert({"aa", 1});

  CachedStringEncoder encoder(vocabulary);

  auto encoded = encoder.EncodeAndCache("aaaaa");
  ASSERT_EQ(2, encoded.size());
  ASSERT_EQ(0, encoded[0]);
  ASSERT_EQ(1, encoded[1]);
}

TEST(CachedStringEncoder, MulticharUnknown) {
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"abc", 0});
  vocabulary.insert({"abd", 1});

  CachedStringEncoder encoder(vocabulary);

  auto encoded = encoder.EncodeAndCache("abe");
  ASSERT_EQ(3, encoded.size());
  ASSERT_EQ(encoder.UnknownElement(), encoded[0]);
  ASSERT_EQ(encoder.UnknownElement(), encoded[1]);
  ASSERT_EQ(encoder.UnknownElement(), encoded[2]);
}

TEST(CachedStringEncoder, MulticharUnknownWithSharedRoot) {
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"abc", 0});
  vocabulary.insert({"abd", 1});
  vocabulary.insert({"ab", 2});

  CachedStringEncoder encoder(vocabulary);

  auto encoded = encoder.EncodeAndCache("abe");
  ASSERT_EQ(2, encoded.size());
  ASSERT_EQ(2, encoded[0]);
  ASSERT_EQ(encoder.UnknownElement(), encoded[1]);
}

TEST(CachedStringEncoder, MulticharUnknownWithSharedSingleChars) {
  // Vocabulary: {"abcd": 0, "a": 1, "b": 2, "c": 3, "e": 4}
  // Input: "abce"
  // Expected behaviour:
  //   1. Forward prefix match "ab".
  //   2. Forward prefix match "abc".
  //   3. No prefix match "abce".
  //   4. No back match "abc".
  //   5. No back match "ab".
  //   6. No back match "abc".
  //   7. No back match "ab".
  //   8. Emit "a" -> 1
  //   9. No prefix match "bc".
  //  10. Emit "b" -> 2.
  //  11. No prefix match "cd".
  //  12. Emit "c" -> 3.
  //  13. No prefix match "e".
  //  14. Emit "c" -> 4.
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"abcd", 0});
  vocabulary.insert({"a", 1});
  vocabulary.insert({"b", 2});
  vocabulary.insert({"c", 3});
  vocabulary.insert({"e", 4});

  CachedStringEncoder encoder(vocabulary);

  auto encoded = encoder.EncodeAndCache("abce");
  ASSERT_EQ(4, encoded.size());
  ASSERT_EQ(1, encoded[0]);
  ASSERT_EQ(2, encoded[1]);
  ASSERT_EQ(3, encoded[2]);
  ASSERT_EQ(4, encoded[3]);
}

}  // namespace
}  // namespace ml4pl

TEST_MAIN();
