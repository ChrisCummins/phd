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
#include "deeplearning/ml4pl/graphs/xla2graph/hlo_module_graph_builder.h"
#include "deeplearning/ml4pl/graphs/xla2graph/xla_stringifier.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status_macros.h"
#include "labm8/cpp/string.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include <sstream>

namespace ml4pl {

labm8::StatusOr<ProgramGraphProto> HloModuleGraphBuilder::Build(
    const xla::HloProto& proto) {
  RETURN_IF_ERROR(VisitModule(proto.hlo_module()));
  return GetGraph();
}

labm8::Status HloModuleGraphBuilder::VisitModule(
    const xla::HloModuleProto& module) {
  // Instantiate the "functions" from HloComputations. Functions are defined in
  // the order of dependencies.
  for (int i = 0; i < module.computations_size(); ++i) {
    FunctionEntryExits computation;
    ASSIGN_OR_RETURN(computation, VisitComputation(module.computations(i)));
    computations_.insert({module.computations(i).id(), computation});
  }

  // Add the call edges from the graph root to the entry computation.
  auto entryComputation = computations_.find(module.entry_computation_id());
  if (entryComputation == computations_.end()) {
    return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                         "Failed to locate entry computation");
  }
  RETURN_IF_ERROR(AddCallEdges(0, entryComputation->second));

  return labm8::Status::OK;
}

labm8::StatusOr<FunctionEntryExits> HloModuleGraphBuilder::VisitComputation(
    const xla::HloComputationProto& computation) {
  int fn;
  ASSIGN_OR_RETURN(fn, AddFunction(computation.name()));

  // Create an entry statement which acts as a common control predecessor of
  // the computation inputs. Since a HLO module is a dataflow graph, there may
  // be multiple inputs.
  int entryInstruction;
  ASSIGN_OR_RETURN(entryInstruction, AddStatement("; computation entry", fn));

  // Visit the instructions in-order. Instructions are ordered such that
  // producers appear before consumers.
  int lastInstruction = entryInstruction;
  for (int i = 0; i < computation.instructions_size(); ++i) {
    ASSIGN_OR_RETURN(
        lastInstruction,
        VisitInstruction(computation.instructions(i), fn, entryInstruction));
  }

  // Since instructions are in a valid execution order, the last instruction
  // must be the final producer.
  return FunctionEntryExits{entryInstruction, {lastInstruction}};
}

labm8::StatusOr<int> HloModuleGraphBuilder::VisitInstruction(
    const xla::HloInstructionProto& instruction, int functionNumber,
    int entryInstruction) {
  // Generate the instruction node.
  int instructionNodeId;
  ASSIGN_OR_RETURN(
      instructionNodeId,
      AddStatement(HloInstructionToText(instruction), functionNumber));
  instructions_.insert({instruction.id(), instructionNodeId});

  // Generate the identifier node for the data produced by the instruction.
  int dataId;
  ASSIGN_OR_RETURN(
      dataId,
      AddIdentifier(ShapeProtoToString(instruction.shape()), functionNumber));
  producers_.insert({instruction.id(), dataId});
  AddDataEdge(instructionNodeId, dataId, 0);

  if (instruction.opcode() == "parameter") {
    // Add the implicit control edge from computation entry point to parameters.
    RETURN_IF_ERROR(AddControlEdge(entryInstruction, instructionNodeId));
  } else if (instruction.opcode() == "constant") {
    // Generate the immediate value nodes for constants.
    int literal;
    ASSIGN_OR_RETURN(literal,
                     AddImmediate(LiteralProtoToText(instruction.literal())));
    AddDataEdge(literal, instructionNodeId, instruction.operand_ids_size());
  }

  // Add data and control edges from consumer to producer..
  for (int i = 0; i < instruction.operand_ids_size(); ++i) {
    auto operandData = producers_.find(instruction.operand_ids(i));
    if (operandData == producers_.end()) {
      std::stringstream err;
      err << "Failed to find operand data " << instruction.id() << " <- "
          << instruction.operand_ids(i);
      return labm8::Status(labm8::error::Code::INVALID_ARGUMENT, err.str());
    }
    AddDataEdge(operandData->second, instructionNodeId, i);

    auto pred = instructions_.find(instruction.operand_ids(i));
    if (pred == instructions_.end()) {
      return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                           "Failed to find operand instruction");
    }
    RETURN_IF_ERROR(AddControlEdge(pred->second, instructionNodeId));
  }

  // Add explicit control dependencies.
  for (int i = 0; i < instruction.control_predecessor_ids_size(); ++i) {
    auto pred = instructions_.find(instruction.control_predecessor_ids(i));
    if (pred == instructions_.end()) {
      return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                           "Failed to find control predecessor");
    }
    RETURN_IF_ERROR(AddControlEdge(pred->second, instructionNodeId));
  }

  // Add call edges from instructions to computations.
  for (int i = 0; i < instruction.called_computation_ids_size(); ++i) {
    labm8::int64 calledComputation = instruction.called_computation_ids(i);

    auto calledComputationEntryExits = computations_.find(calledComputation);
    if (calledComputationEntryExits == computations_.end()) {
      return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                           "Failed to locate called computation");
    }
    AddCallEdges(instructionNodeId, calledComputationEntryExits->second);
  }

  return instructionNodeId;
}

}  // namespace ml4pl
