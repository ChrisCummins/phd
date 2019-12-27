// Copyright 2019 the ProGraML authors.
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
#include "deeplearning/ml4pl/graphs/graph_builder.h"

#include "labm8/cpp/test.h"

namespace ml4pl {
namespace {

TEST(GraphBuilder, AddFunction) {
  GraphBuilder builder;
  auto foo = builder.AddFunction("foo");
  ASSERT_EQ(foo.first, 0);
  ASSERT_EQ(foo.second->name(), "foo");

  auto bar = builder.AddFunction("bar");
  ASSERT_EQ(bar.first, 1);
  ASSERT_EQ(bar.second->name(), "bar");
}

TEST(GraphBuilder, AddFunctionWithEmptyName) {
  GraphBuilder builder;
  ASSERT_DEATH(builder.AddFunction(""), "Empty function name is invalid");
}

TEST(GraphBuilder, UnconnectedNode) {
  GraphBuilder builder;
  auto fn = builder.AddFunction("x");
  builder.AddStatement("x", fn.first);
  ASSERT_DEATH(builder.GetGraph(), "Graph contains node with no connections");
}

}  // namespace
}  // namespace ml4pl

TEST_MAIN();
