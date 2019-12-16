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

  auto encoded_1 = encoder.Encode("");
  ASSERT_EQ(0, encoded_1.size());
}

TEST(CachedStringEncoder, EncodeUnknownElements) {
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"a", 0});

  CachedStringEncoder encoder(vocabulary);

  auto encoded_1 = encoder.Encode("bc");
  ASSERT_EQ(2, encoded_1.size());
  ASSERT_EQ(encoder.UnknownElement(), encoded_1[0]);
  ASSERT_EQ(encoder.UnknownElement(), encoded_1[1]);
}

TEST(CachedStringEncoder, EncodeMultiCharMatch) {
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"int", 0});

  CachedStringEncoder encoder(vocabulary);

  auto encoded = encoder.Encode("int");
  ASSERT_EQ(1, encoded.size());
  ASSERT_EQ(0, encoded[0]);
}

TEST(CachedStringEncoder, EncodeMultiCharMatchComposite) {
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"int", 0});
  vocabulary.insert({" ", 1});

  CachedStringEncoder encoder(vocabulary);

  auto encoded = encoder.Encode("int int");
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
  auto encoded_1 = encoder.Encode("a");
  ASSERT_EQ(1, encoded_1.size());
  ASSERT_EQ(0, encoded_1[0]);

  // Test encoding the same input again.
  auto encoded_2 = encoder.Encode("a");
  ASSERT_EQ(1, encoded_2.size());
  ASSERT_EQ(0, encoded_2[0]);
}

TEST(CachedStringEncoder, UnknownFinalToken) {
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"int", 0});

  CachedStringEncoder encoder(vocabulary);

  auto encoded = encoder.Encode("int ");
  ASSERT_EQ(2, encoded.size());
  ASSERT_EQ(0, encoded[0]);
  ASSERT_EQ(encoder.UnknownElement(), encoded[1]);
}

TEST(CachedStringEncoder, UnknownFinalTokens) {
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"int", 0});

  CachedStringEncoder encoder(vocabulary);

  auto encoded = encoder.Encode("int   ");
  ASSERT_EQ(4, encoded.size());
  ASSERT_EQ(0, encoded[0]);
  ASSERT_EQ(encoder.UnknownElement(), encoded[1]);
  ASSERT_EQ(encoder.UnknownElement(), encoded[2]);
  ASSERT_EQ(encoder.UnknownElement(), encoded[3]);
}

TEST(CachedStringEncoder, MulticharUnknown) {
  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"abc", 0});
  vocabulary.insert({"abd", 1});

  CachedStringEncoder encoder(vocabulary);

  auto encoded = encoder.Encode("abe");
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

  auto encoded = encoder.Encode("abe");
  ASSERT_EQ(2, encoded.size());
  ASSERT_EQ(2, encoded[0]);
  ASSERT_EQ(encoder.UnknownElement(), encoded[1]);
}

}  // namespace
}  // namespace ml4pl

TEST_MAIN();
