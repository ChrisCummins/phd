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
#include "programl/graph/analysis/reachability.h"

#include "labm8/cpp/status.h"
#include "labm8/cpp/test.h"
#include "programl/graph/program_graph_builder.h"

using labm8::Status;

namespace programl {
namespace graph {
namespace analysis {
namespace {

TEST(ReachabilityAnalysis, Foo) {
  ProgramGraphBuilder builder;
  Module* mod = builder.AddModule("mod");
  Function* fn = builder.AddFunction("fn", mod);
  Node* a = builder.AddInstruction("a", fn);
  Node* b = builder.AddInstruction("b", fn);
  ASSERT_OK(builder.AddControlEdge(0, builder.GetRootNode(), a));
  ASSERT_OK(builder.AddControlEdge(0, a, b));
  ProgramGraph graph = builder.Build().ValueOrDie();

  ReachabilityAnalysis analysis(graph);
  ProgramGraphFeaturesList features;
  ASSERT_OK(analysis.Run(&features));
  ASSERT_EQ(features.graph_size(), 1);
}

}  // anonymous namespace
}  // namespace analysis
}  // namespace graph
}  // namespace programl

TEST_MAIN();