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
#include "programl/graph/analysis/liveness.h"

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/test.h"
#include "programl/graph/program_graph_builder.h"
#include "programl/test/analysis_testutil.h"

using labm8::Status;

namespace programl {
namespace graph {
namespace analysis {
namespace {

class LivenessAanalysisTest : public labm8::Test {
 public:
  LivenessAanalysisTest() {
    // Example graph taken from Wikipedia:
    // https://en.wikipedia.org/wiki/Live_variable_analysis
    //
    // // in: {}
    // b1: a = 3;
    //     b = 5;
    //     d = 4;
    //     x = 100; //x is never being used later thus not in the out set
    //     {a,b,d} if a > b then
    // // out: {a,b,d}    //union of all (in) successors of b1 => b2: {a,b}, and
    // b3:{b,d}
    //
    // // in: {a,b}
    // b2: c = a + b;
    //     d = 2;
    // // out: {b,d}
    //
    // // in: {b,d}
    // b3: endif
    //     c = 4;
    //     return b * d + c;
    // // out:{}
    ProgramGraphBuilder builder;
    auto mod = builder.AddModule("mod");
    auto fn = builder.AddFunction("fn", mod);

    // Variables.
    auto a = builder.AddVariable("a", fn);  // 1
    auto b = builder.AddVariable("b", fn);  // 2
    auto c = builder.AddVariable("c", fn);  // 3
    auto d = builder.AddVariable("d", fn);  // 4
    auto x = builder.AddVariable("x", fn);  // 5

    // Blocks.
    auto b1 = builder.AddInstruction("b1", fn);    // 6
    auto b2 = builder.AddInstruction("b2", fn);    // 7
    auto b3a = builder.AddInstruction("b3a", fn);  // 8
    auto b3b = builder.AddInstruction("b3b", fn);  // 9

    // Control edges.
    CHECK(builder.AddControlEdge(0, builder.GetRootNode(), b1).ok());
    CHECK(builder.AddControlEdge(0, b1, b2).ok());
    CHECK(builder.AddControlEdge(0, b1, b3a).ok());
    CHECK(builder.AddControlEdge(0, b2, b3a).ok());
    CHECK(builder.AddControlEdge(0, b3a, b3b).ok());

    // Defs.
    CHECK(builder.AddDataEdge(0, b1, a).ok());
    CHECK(builder.AddDataEdge(0, b1, b).ok());
    CHECK(builder.AddDataEdge(0, b1, d).ok());
    CHECK(builder.AddDataEdge(0, b1, x).ok());
    CHECK(builder.AddDataEdge(0, b2, c).ok());
    CHECK(builder.AddDataEdge(0, b2, d).ok());
    CHECK(builder.AddDataEdge(0, b3a, c).ok());

    // Uses.
    CHECK(builder.AddDataEdge(0, a, b2).ok());
    CHECK(builder.AddDataEdge(0, b, b2).ok());
    CHECK(builder.AddDataEdge(0, b, b3b).ok());
    CHECK(builder.AddDataEdge(0, c, b3b).ok());
    CHECK(builder.AddDataEdge(0, d, b3b).ok());

    wiki_ = builder.Build().ValueOrDie();
  }

 protected:
  ProgramGraph wiki_;
};

TEST_F(LivenessAanalysisTest, WikiWithRootB1) {
  LivenessAnalysis analysis(wiki_);
  analysis.Init();
  ProgramGraphFeatures g;

  ASSERT_OK(analysis.RunOne(6, &g));

  // EXPECT_ACTIVE_NODE_COUNT(g, 3);
  EXPECT_STEP_COUNT(g, 5);

  // Features.
  EXPECT_NOT_ROOT(g, 0);  // <root>
  EXPECT_NOT_ROOT(g, 1);  // a
  EXPECT_NOT_ROOT(g, 2);  // b
  EXPECT_NOT_ROOT(g, 3);  // c
  EXPECT_NOT_ROOT(g, 4);  // d
  EXPECT_NOT_ROOT(g, 5);  // x
  EXPECT_ROOT(g, 6);      // b1
  EXPECT_NOT_ROOT(g, 7);  // b2
  EXPECT_NOT_ROOT(g, 8);  // b3a
  EXPECT_NOT_ROOT(g, 9);  // b3b

  // Labels.
  EXPECT_NODE_FALSE(g, 0);  // <root>
  EXPECT_NODE_TRUE(g, 1);   // a
  EXPECT_NODE_TRUE(g, 2);   // b
  EXPECT_NODE_FALSE(g, 3);  // c
  EXPECT_NODE_TRUE(g, 4);   // d
  EXPECT_NODE_FALSE(g, 5);  // x
  EXPECT_NODE_FALSE(g, 6);  // b1
  EXPECT_NODE_FALSE(g, 7);  // b2
  EXPECT_NODE_FALSE(g, 8);  // b3a
  EXPECT_NODE_FALSE(g, 9);  // b3b
}

// TEST_F(LivenessAanalysisTest, WikiWithRootB2) {
//  LivenessAnalysis analysis(wiki_);
//  analysis.Init();
//  ProgramGraphFeatures g;
//
//  ASSERT_OK(analysis.RunOne(7, &g));
//  EXPECT_ACTIVE_NODE_COUNT(g, 2);
//  EXPECT_STEP_COUNT(g, 5);
//
//  // Features.
//  EXPECT_NOT_ROOT(g, 0);  // <root>
//  EXPECT_NOT_ROOT(g, 1);  // a
//  EXPECT_NOT_ROOT(g, 2);  // b
//  EXPECT_NOT_ROOT(g, 3);  // c
//  EXPECT_NOT_ROOT(g, 4);  // d
//  EXPECT_NOT_ROOT(g, 5);  // x
//  EXPECT_NOT_ROOT(g, 6);  // b1
//  EXPECT_ROOT(g, 7);  // b2
//  EXPECT_NOT_ROOT(g, 8);  // b3a
//  EXPECT_NOT_ROOT(g, 9);  // b3b
//
//  // Labels.
//  EXPECT_NODE_FALSE(g, 0);  // <root>
//  EXPECT_NODE_FALSE(g, 1);  // a
//  EXPECT_NODE_TRUE(g, 2);  // b
//  EXPECT_NODE_FALSE(g, 3);  // c
//  EXPECT_NODE_TRUE(g, 4);  // d
//  EXPECT_NODE_FALSE(g, 5);  // x
//  EXPECT_NODE_FALSE(g, 6);  // b1
//  EXPECT_NODE_FALSE(g, 7);  // b2
//  EXPECT_NODE_FALSE(g, 8);  // b3a
//  EXPECT_NODE_FALSE(g, 9);  // b3b
//}

}  // anonymous namespace
}  // namespace analysis
}  // namespace graph
}  // namespace programl

TEST_MAIN();