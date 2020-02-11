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

#include "deeplearning/ml4pl/seq/graph_serializer.h"
#include "labm8/cpp/logging.h"

#include <vector>

namespace ml4pl {

const string& GetString(const ProgramGraphProto& graph, int index) {
  CHECK(index >= 0 && index <= graph.string_size())
      << "String " << index << " is out of range for string table with "
      << graph.string_size() << " elements";
  return graph.string(index);
}

ProgramGraphSeqProto GraphEncoder::Encode(const ProgramGraphProto& graph) {
  ProgramGraphSeqProto message;

  auto encoded = message.mutable_encoded();
  auto encoded_node_lengths = message.mutable_encoded_node_length();
  auto nodes = message.mutable_node();

  for (const auto& node : SerializeStatements(graph)) {
    // Encode the text of the node.
    std::vector<int> encoded_node =
        string_encoder_.Encode(GetString(graph, graph.node(node).text()));
    // Append the encoded node and segment IDs.
    for (const auto& x : encoded_node) {
      encoded->Add(x);
    }
    // Record the node index and the size of the encoded array.
    encoded_node_lengths->Add(encoded_node.size());
    nodes->Add(node);
  }

  return message;
}

}  // namespace ml4pl
