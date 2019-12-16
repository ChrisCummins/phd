#include "deeplearning/ml4pl/seq/graph_encoder.h"

#include "deeplearning/ml4pl/seq/graph_serializer.h"

#include <vector>

namespace ml4pl {

ProgramGraphSeq GraphEncoder::Encode(const ProgramGraph& graph) {
  ProgramGraphSeq message;

  auto encoded = message.mutable_encoded();
  auto encoded_node_lengths = message.mutable_encoded_node_length();
  auto nodes = message.mutable_node();

  for (const auto& node : SerializeStatements(graph)) {
    // Encode the text of the node.
    std::vector<int> encoded_node =
        string_encoder_.EncodeAndCache(graph.node(node).text());
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
