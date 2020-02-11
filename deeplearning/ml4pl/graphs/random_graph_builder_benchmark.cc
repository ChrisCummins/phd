#include "deeplearning/ml4pl/graphs/random_graph_builder.h"

#include "labm8/cpp/test.h"

namespace ml4pl {

namespace {

static void BM_GenerateRandomProgramGraph(benchmark::State &state) {
  int nodeCount = static_cast<int>(state.range(0));
  while (state.KeepRunning()) {
    RandomGraphBuilder builder;
    auto graph = builder.FastCreateRandom(nodeCount).ValueOrDie();
    benchmark::DoNotOptimize(graph);
  }
}
BENCHMARK(BM_GenerateRandomProgramGraph)->Range(10, 1000);

}  // anonymous namespace

}  // namespace ml4pl

TEST_MAIN();
