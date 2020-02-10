// A class for constructing program graphs from LLVM modules.
//
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
#pragma once

#include <vector>

#include "absl/container/flat_hash_map.h"

#include "deeplearning/ml4pl/graphs/graph_builder.h"
#include "deeplearning/ml4pl/graphs/programl.pb.h"
#include "labm8/cpp/port.h"
#include "labm8/cpp/statusor.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"

namespace ml4pl {

// A class for generating program graphs from HloProto messages.
class HloModuleGraphBuilder : GraphBuilder {
 public:
  // Main entry point. Accepts a module as input and returns a graph as output,
  // or an error status if graph construction fails.
  labm8::StatusOr<ProgramGraphProto> Build(const xla::HloProto& proto);

 protected:
  labm8::Status VisitModule(const xla::HloModuleProto& module);

  labm8::StatusOr<FunctionEntryExits> VisitComputation(
      const xla::HloComputationProto& computation);

  labm8::StatusOr<size_t> VisitInstruction(
      const xla::HloInstructionProto& instruction, size_t functionNumber,
      size_t entryInstruction);

 private:
  // A map from computations to their entry and exit nodes.
  absl::flat_hash_map<labm8::int64, FunctionEntryExits> computations_;
  // A map of instruction IDs to their node number.
  absl::flat_hash_map<labm8::int64, size_t> instructions_;
  // A map of instruction IDs to the data element produced by the instruction.
  absl::flat_hash_map<labm8::int64, size_t> producers_;
};

}  // namespace ml4pl
