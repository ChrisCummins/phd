#include "deeplearning/ml4pl/seq/graph_encoder.h"

#include "deeplearning/ml4pl/seq/cached_string_encoder.h"
#include "deeplearning/ml4pl/seq/graph2seq.pb.h"

#include "absl/container/flat_hash_map.h"

#include "labm8/cpp/test.h"

namespace ml4pl {
namespace {

TEST(GraphEncoder, EmptyGraph) {
  ProgramGraph graph;

  absl::flat_hash_map<string, int> vocabulary;
  vocabulary.insert({"a", 0});
  CachedStringEncoder string_encoder(vocabulary);
  GraphEncoder encoder(string_encoder);

  auto encoded = encoder.Encode(graph);

  ASSERT_EQ(0, encoded.encoded_size());
  ASSERT_EQ(0, encoded.encoded_node_length_size());
  ASSERT_EQ(0, encoded.node_size());
}

}  // namespace
}  // namespace ml4pl

TEST_MAIN();
