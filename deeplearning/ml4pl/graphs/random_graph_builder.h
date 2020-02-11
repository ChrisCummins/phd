#pragma once

#include <time.h>
#include "deeplearning/ml4pl/graphs/graph_builder.h"
#include "deeplearning/ml4pl/graphs/programl.pb.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/statusor.h"

using labm8::Status;
using labm8::StatusOr;
namespace error = labm8::error;

namespace ml4pl {

class RandomGraphBuilder : public GraphBuilder {
 public:
  RandomGraphBuilder(int seed = 0) : GraphBuilder() {
    if (!seed) {
      srand(time(NULL));
    } else {
      srand(seed);
    }
  }

  // Generates a random graph which has sensible values for fields, but does not
  // have meaningful semantics, e.g. there may be data flow edges between
  // identifiers, etc. For speed, this generator guarantees only that:
  //
  //   1. There is a 'root' node with outgoing call edges.
  //   2. Nodes are either statements, identifiers, or immediates.
  //   3. Nodes have text, preprocessed_text, and a single node_x value.
  //   4. Edges are either control, data, or call.
  //   5. Edges have positions.
  //   6. The graph is strongly connected.
  StatusOr<ProgramGraphProto> FastCreateRandom(int nodeCount);
};

}  // namespace ml4pl
