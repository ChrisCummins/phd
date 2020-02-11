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
#include "deeplearning/ml4pl/graphs/graph_builder.h"

#include "labm8/cpp/logging.h"
#include "labm8/cpp/test.h"

namespace ml4pl {
namespace {

const string& GetString(const ProgramGraphProto& graph, int index) {
  CHECK(index >= 0 && index <= graph.string_size())
      << "String " << index << " is out of range for string table with "
      << graph.string_size() << " elements";
  return graph.string(index);
}

TEST(GraphBuilder, AddFunction) {
  GraphBuilder builder;
  int foo = builder.AddFunction("foo").ValueOrDie();
  int bar = builder.AddFunction("bar").ValueOrDie();

  ASSERT_EQ(foo, 0);
  ASSERT_EQ(bar, 1);

  ProgramGraphProto graph = builder.GetGraph().ValueOrDie();
  ASSERT_EQ(GetString(graph, graph.function(foo).name()), "foo");
  ASSERT_EQ(GetString(graph, graph.function(bar).name()), "bar");
}

TEST(GraphBuilder, AddFunctionWithEmptyName) {
  GraphBuilder builder;
  StatusOr<int> fn = builder.AddFunction("");
  ASSERT_FALSE(fn.ok());
  ASSERT_EQ(fn.status().error_message(), "Empty function name is invalid");
}

TEST(GraphBuilder, UnconnectedNode) {
  GraphBuilder builder;
  auto fn = builder.AddFunction("x").ValueOrDie();
  builder.AddStatement("x", fn);
  ASSERT_FALSE(builder.GetGraph().ok());
}

}  // namespace
}  // namespace ml4pl

TEST_MAIN();
