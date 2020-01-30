// Copyright 2019-2020 the ProGraML authors.
//
// Contact Chris Cummins <chrisc.101@gmail.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
