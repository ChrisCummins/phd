#include "deeplearning/ml4pl/graphs/random_graph_builder.h"
#include <cstdlib>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "fmt/format.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/status_macros.h"

namespace ml4pl {

// Generate a random integer in the range min <= x <= max.
template <typename T = int>
T RandInt(T min, T max) {
  if (min == max) {
    return min;
  }
  CHECK(min < max) << "RandInt(" << min << ", " << max
                   << ") values are illegal";
  return min + (rand() % (max - min + 1));
}

// Generate a random floating point number in the range 0 <= x <= 1.
template <typename T = float>
T RandFloat(T min, T max) {
  if (min == max) {
    return min;
  }
  CHECK(min < max) << "RandFloat(" << min << ", " << max
                   << ") values are illegal";
  return min +
         static_cast<T>(rand()) / (static_cast<T>(RAND_MAX / (max - min)));
}

size_t FlatMatrixIndex(int y, int x, int rowLength) {
  return rowLength * y + x;
}

// Time: O(nodeCount)
// Space: O(nodeCount ^ 2)
StatusOr<ProgramGraphProto> RandomGraphBuilder::FastCreateRandom(
    int nodeCount) {
  if (!nodeCount) {
    nodeCount = RandInt(15, 50);
  }

  if (nodeCount < 2 || nodeCount > 1000) {
    return Status(error::Code::INVALID_ARGUMENT, "Illegal node count: {}",
                  nodeCount);
  }

  // Divide node count amongst statements / immediates / identifiers.
  int remainingNodes = nodeCount;
  int statementCount =
      std::max(1, static_cast<int>(RandFloat(0.6, 0.8) * remainingNodes));
  remainingNodes -= statementCount;
  int identifierCount =
      std::max(1, static_cast<int>(RandFloat(0.7, 0.8) * remainingNodes));
  int immediatesCount = nodeCount - 1 - statementCount - identifierCount;

  LOG(INFO) << "Node counts: " << statementCount << ", " << identifierCount
            << ", " << immediatesCount;

  CHECK(identifierCount + immediatesCount + statementCount + 1 == nodeCount)
      << identifierCount << " + " << immediatesCount << " + " << statementCount
      << " != " << nodeCount;

  // Generate the nodes.
  int function = 0;

  // A list of maximum function.
  std::vector<int> functions;

  // A flattened adjacency matrix used to prevent parallel control edges.
  std::vector<bool> backwardControlEdges(statementCount * statementCount,
                                         false);

  for (int i = 0; i < statementCount; ++i) {
    int node = NextNodeNumber();

    // Maybe create a new function.
    if (!functions.size() ||
        (node - functions[functions.size() - 1] > 1 && RandInt(0, 100) > 85)) {
      ASSIGN_OR_RETURN(function,
                       AddFunction(fmt::format("func.{}", FunctionCount())));

      functions.push_back(node - 1);

      // When generating new statements, we create only control edges from
      // predecessor to successor. Once we have generated all of the nodes in
      // a function we can now create some back edges.
      int forwardEdgeCount =
          (functions[functions.size() - 1] - functions[functions.size() - 2]);
      if (forwardEdgeCount > 1) {
        int backEdgeCount = RandInt(0, forwardEdgeCount - 1);
        for (int j = 0; j < backEdgeCount; ++j) {
          int pred = RandInt(functions[functions.size() - 2] + 1,
                             functions[functions.size() - 1] - 1);
          int succ = RandInt(pred + 1, functions[functions.size() - 1]);

          size_t backwardEdge = FlatMatrixIndex(succ, pred, statementCount);
          if (!backwardControlEdges[backwardEdge]) {
            RETURN_IF_ERROR(AddControlEdge(succ, pred));
            backwardControlEdges[backwardEdge] = true;
          }
        }
      }
    }

    ASSIGN_OR_RETURN(node, AddStatement("stmt", function));

    if (node == functions[functions.size() - 1] + 1) {
      // New function. Pick a calling instruction from anywhere in the set of
      // statements.
      int predNode = node > 1 ? RandInt(0, node - 1) : 0;
      RETURN_IF_ERROR(AddCallEdge(predNode, node));
    } else {
      // Within a function.
      int predNode = RandInt(functions[functions.size() - 1] + 1, node - 1);
      RETURN_IF_ERROR(AddControlEdge(predNode, node));
    }
  }

  functions.push_back(NextNodeNumber() - 1);

  // A map from statement ID to next available position number. Incremented
  // every time a new data edge is added.
  std::vector<int> dataPositions(nodeCount, 0);
  std::vector<bool> identifierEdges(nodeCount * nodeCount, false);
  std::vector<bool> immediateEdges(nodeCount * nodeCount, false);

  // Add the identifiers.
  for (int i = 0; i < identifierCount; ++i) {
    size_t functionIdx = RandInt<size_t>(1, functions.size() - 1);

    // Get the range of nodes for a function.
    int nodeStart = functions[functionIdx - 1] + 1;
    int nodeEnd = functions[functionIdx];
    int numNodes = nodeEnd - nodeStart;

    if (!numNodes) {
      continue;
    }

    int function = GetNode(functions[functionIdx]).function();

    // Create the identifier.
    int identifier;
    ASSIGN_OR_RETURN(identifier, AddIdentifier("ident", function));

    for (int j = 0; j < RandInt(1, std::min(5, nodeEnd - nodeStart)); ++j) {
      int statement = RandInt(nodeStart, nodeEnd);

      int src, dst;
      if (RandInt(0, 10) < 5) {
        src = statement;
        dst = identifier;
      } else {
        src = identifier;
        dst = statement;
      }

      int position = dataPositions[dst];
      ++dataPositions[dst];

      size_t dataEdge = FlatMatrixIndex(src, dst, nodeCount);
      if (!identifierEdges[dataEdge]) {
        RETURN_IF_ERROR(AddDataEdge(src, dst, position));
        identifierEdges[dataEdge] = true;
      }
    }
  }

  // Add the immediates.
  for (int i = 0; i < immediatesCount; ++i) {
    int immediate;
    ASSIGN_OR_RETURN(immediate, AddImmediate("const"));

    for (int j = 0;
         j < std::min(5, RandInt(1, std::max(statementCount / 10, 1))); ++j) {
      int dst = RandInt(1, statementCount);
      int position = dataPositions[dst];
      ++dataPositions[dst];

      size_t dataEdge = FlatMatrixIndex(immediate, dst, nodeCount);
      if (!immediateEdges[dataEdge]) {
        RETURN_IF_ERROR(AddDataEdge(immediate, dst, position));
        immediateEdges[dataEdge] = true;
      }
    }
  }

  CHECK(NextNodeNumber() == nodeCount) << "NextNodeNumber " << NextNodeNumber();

  return GetGraph();
}

}  // namespace ml4pl
