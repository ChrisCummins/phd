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
#include "deeplearning/ml4pl/graphs/llvm2graph/llvm_graph_builder.h"

#include <deque>
#include <sstream>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "deeplearning/ml4pl/graphs/programl.pb.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status_macros.h"
#include "labm8/cpp/string.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"

namespace ml4pl {

// Produce the textual representation of an LLVM value.
// This accepts instances of llvm::Instruction or llvm::Value.
template <typename T>
string PrintToString(const T& value) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  value.print(rso);
  // Trim any leading indentation whitespace.
  labm8::TrimLeft(str);
  return str;
}

StatusOr<string> GetInstructionLhs(const llvm::Instruction& instruction) {
  const string instructionString = PrintToString(instruction);
  const size_t snipAt = instructionString.find(" = ");
  if (snipAt == string::npos) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "' = ' assignment not found in instruction: {}",
                  instructionString);
  }

  const string identifier = instructionString.substr(0, snipAt);
  const string type = PrintToString(*instruction.getType());

  std::stringstream instructionName;
  instructionName << type << ' ' << identifier;
  return instructionName.str();
}

StatusOr<string> GetInstructionRhs(const llvm::Instruction& instruction) {
  const string instructionString = PrintToString(instruction);
  const size_t snipAt = instructionString.find(" = ");
  if (snipAt == string::npos) {
    return instructionString;
  }

  return instructionString.substr(snipAt + 3);
}

StatusOr<BasicBlockEntryExit> LlvmGraphBuilder::VisitBasicBlock(
    const llvm::BasicBlock& block, const int& functionNumber,
    InstructionNumberMap* instructions, ArgumentConsumerMap* argumentConsumers,
    std::vector<DataEdge>* dataEdgesToAdd) {
  // Keep track of the basic block entry number and the current node number.
  int firstNodeNumber = NextNodeNumber();
  int currentNodeNumber = -1;
  int previousNodeNumber = -1;

  // Iterate over the instructions of a basic block in-order.
  for (const llvm::Instruction& instruction : block) {
#ifdef PROGRAML_FUTURE_NODE_REPRESENTATION
    // TODO(github.com/ChrisCummins/ProGraML/issues/55): Don't use the entire
    // text of an instruction (e.g. "%3 = add %1 %2") for statement nodes.
    const string text = GetInstructionRhs(instruction);
#else
    const string text = PrintToString(instruction);
#endif

    // Create the graph node for the instruction.
    int statement;
    ASSIGN_OR_RETURN(statement, AddStatement(text, functionNumber));
    previousNodeNumber = currentNodeNumber;
    currentNodeNumber = statement;

    instructions->insert({&instruction, currentNodeNumber});

    // A basic block consists of a linear sequence of instructions, so we can
    // insert the control edge between instructions as we go.
    if (previousNodeNumber != -1) {
      RETURN_IF_ERROR(AddControlEdge(previousNodeNumber, currentNodeNumber));
    }

    // If the instruction is a call, record the call site, which we will use
    // later to create all edges..
    if (auto* callInstruction = llvm::dyn_cast<llvm::CallInst>(&instruction)) {
      auto calledFunction = callInstruction->getCalledFunction();
      // TODO(github.com/ChrisCummins/ProGraML/issues/46): Should we handle the
      // case when getCalledFunction() is nil?
      if (calledFunction) {
        call_sites_.insert({currentNodeNumber, calledFunction});
      }
    }

    // Iterate over the usedef chains by reading operands of the currrent
    // instruction.
    int position = 0;
    for (const llvm::Use& use : instruction.operands()) {
      const llvm::Value* value = use.get();

      if (llvm::dyn_cast<llvm::Function>(value)) {
        // Functions are a subclass of llvm::Constant, but we do not want to
        // treat them the same way as other constants (i.e. generate data
        // elements for them). Instead we ignore them, as the call edges that
        // will be produced provide the information we want to capture.
      } else if (const auto* constant = llvm::dyn_cast<llvm::Constant>(value)) {
        // If the operand is a constant value, insert a new entry into the map
        // of constants to node IDs and positions. We defer creating the
        // immediate nodes until we have traversed all
        constants_[constant].push_back({currentNodeNumber, position});
      } else if (const auto* operand =
                     llvm::dyn_cast<llvm::Instruction>(value)) {
#ifdef PROGRAML_FUTURE_NODE_REPRESENTATION
        // TODO(github.com/ChrisCummins/ProGraML/issues/55): Set name of the
        // identifier to the LHS of the instruction.
        const string identifierText = GetInstructionLhs(*operand);
#else
        const string identifierText = "!IDENTIFIER";
#endif

        // We have an instruction operand which itself is another instruction.
        //
        // For example, take the following IR snippet:
        //
        //     %2 = load i32, i32* %1, align 4
        //     %3 = add nsw i32 %2, 1
        //
        // Here, instruction '%3' uses the result of '%2' as an operand, so we
        // want to construct a graph like:
        //
        //     STATEMENT:  load i32, i32* %1, align 4
        //                      |
        //                      V
        //     IDENTIFIER:      %2
        //                      |
        //                      V
        //     STATEMENT:  %3 = add nsw i32 %2, 1
        //
        // To this we create the intermediate data flow node '%2' immediately,
        // but defer adding the edge from the producer instruction, since we may
        // not have visited it yet.
        int identifier;
        ASSIGN_OR_RETURN(identifier,
                         AddIdentifier(identifierText, functionNumber));

        // Connect the data -> consumer.
        RETURN_IF_ERROR(AddDataEdge(identifier, currentNodeNumber, position));

        // Defer creation of the edge from producer -> data.
        dataEdgesToAdd->push_back({operand, identifier});
      } else if (const auto* operand = llvm::dyn_cast<llvm::Argument>(value)) {
        // Record the usage of the argument.
        (*argumentConsumers)[operand].push_back({currentNodeNumber, position});
      } else if (!(
                     // Basic blocks are not considered data. There will instead
                     // be a control edge from this instruction to the entry
                     // node of the block.
                     llvm::dyn_cast<llvm::BasicBlock>(value) ||
                     // Inline assembly is ignored.
                     llvm::dyn_cast<llvm::InlineAsm>(value) ||
                     // Nothing to do here.
                     llvm::dyn_cast<llvm::MetadataAsValue>(value))) {
        LOG(FATAL) << "Unknown operand " << position << " for instruction:\n  "
                   << "\n  Instruction:  " << PrintToString(instruction)
                   << "\n  Operand:      " << PrintToString(*value)
                   << "\n  Operand Type: " << PrintToString(*value->getType());
      }

      // Advance to the next operand position.
      ++position;
    }
  }

  if (currentNodeNumber == -1) {
    return Status(error::Code::INVALID_ARGUMENT, "No instructions in block");
  }

  return std::make_pair(firstNodeNumber, currentNodeNumber);
}

StatusOr<FunctionEntryExits> LlvmGraphBuilder::VisitFunction(
    const llvm::Function& function, const int& functionNumber) {
  // A map from basic blocks to <entry,exit> node numbers.
  absl::flat_hash_map<const llvm::BasicBlock*, BasicBlockEntryExit> blocks;
  // A map from function Arguments to the statements that consume them, and the
  // position of the argument in the statement operand list.
  ArgumentConsumerMap argumentConsumers;
  // A map of instruction numbers which will be used to resolve the node numbers
  // for inter-instruction data flow edges once all basic blocks have been
  // visited.
  InstructionNumberMap instructions;

  // A mapping from producer instructions to consumer instructions.
  std::vector<DataEdge> dataEdgesToAdd;

  FunctionEntryExits functionEntryExits;

  if (function.isDeclaration()) {
    int entry, exit;
    ASSIGN_OR_RETURN(
        entry, AddStatement("; undefined function entry", functionNumber));
    ASSIGN_OR_RETURN(exit,
                     AddStatement("; undefined function exit", functionNumber));
    RETURN_IF_ERROR(AddControlEdge(entry, exit));
    functionEntryExits.first = entry;
    functionEntryExits.second.push_back(exit);
    return functionEntryExits;
  }

  // Visit all basic blocks in the function to construct the per-block graph
  // components.
  for (const llvm::BasicBlock& block : function) {
    std::pair<int, int> entry_exit;
    ASSIGN_OR_RETURN(entry_exit,
                     VisitBasicBlock(block, functionNumber, &instructions,
                                     &argumentConsumers, &dataEdgesToAdd));
    blocks.insert({&block, entry_exit});
  }
  if (!blocks.size()) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "Function contains no blocks: {}",
                  string(function.getName()));
  }

  // Construct the identifier data elements for arguments and connect data
  // edges.
  for (auto it : argumentConsumers) {
#ifdef PROGRAML_FUTURE_NODE_REPRESENTATION
    auto text = PrintToString(*it.first);
#else
    auto text = "!IDENTIFIER";
#endif
    int argument;
    ASSIGN_OR_RETURN(argument, AddIdentifier(text, functionNumber));
    for (auto argumentConsumer : it.second) {
      RETURN_IF_ERROR(AddDataEdge(argument, argumentConsumer.first,
                                  argumentConsumer.second));
    }
  }

  // Construct the data edges from producer instructions to the data flow
  // elements that are produced.
  for (auto dataEdgeToAdd : dataEdgesToAdd) {
    auto producer = instructions.find(dataEdgeToAdd.first);
    if (producer == instructions.end()) {
      return Status(
          error::Code::INVALID_ARGUMENT,
          "Operand references instruction that has not been visited: {}",
          PrintToString(*dataEdgeToAdd.first));
    }
    RETURN_IF_ERROR(
        AddDataEdge(producer->second, dataEdgeToAdd.second, /*position=*/0));
  }

  const llvm::BasicBlock* entry = &function.getEntryBlock();
  if (!entry) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "No entry block for function: {}",
                  string(function.getName()));
  }

  // Construct a the <entry, exits> pair.
  auto entryNode = blocks.find(entry);
  if (entryNode == blocks.end()) {
    return Status(error::Code::INVALID_ARGUMENT, "No entry block");
  }
  functionEntryExits.first = entryNode->second.first;

  // Traverse the basic blocks in the function, creating control edges between
  // them.
  absl::flat_hash_set<const llvm::BasicBlock*> visited{entry};
  std::deque<const llvm::BasicBlock*> q{entry};

  while (q.size()) {
    const llvm::BasicBlock* current = q.front();
    q.pop_front();

    auto it = blocks.find(current);
    if (it == blocks.end()) {
      return Status(error::Code::INVALID_ARGUMENT, "Block not found");
    }
    int currentExit = it->second.second;

    int successorNumber = -1;

    // For each current -> successor pair, construct a control edge from the
    // last instruction in current to the first instruction in successor.
    for (const llvm::BasicBlock* successor : llvm::successors(current)) {
      ++successorNumber;

      auto it = blocks.find(successor);
      if (it == blocks.end()) {
        return Status(error::Code::INVALID_ARGUMENT, "Block not found");
      }
      int successorEntry = it->second.first;

      RETURN_IF_ERROR(AddControlEdge(currentExit, successorEntry));

      if (visited.find(successor) == visited.end()) {
        q.push_back(successor);
        visited.insert(successor);
      }
    }

    // If the block has no successors, record the block exit instruction.
    if (successorNumber == -1) {
      functionEntryExits.second.push_back(currentExit);
    }
  }
  if (visited.size() != function.getBasicBlockList().size()) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "Visited {} blocks in a function with blocks", visited.size(),
                  function.getBasicBlockList().size());
  }

  return functionEntryExits;
}

StatusOr<ProgramGraphProto> LlvmGraphBuilder::Build(
    const llvm::Module& module) {
  // A map from functions to their entry and exit nodes.
  absl::flat_hash_map<const llvm::Function*, FunctionEntryExits> functions;

  for (const llvm::Function& function : module) {
    // Create the function message.
    int fn;
    ASSIGN_OR_RETURN(fn, AddFunction(function.getName()));

    FunctionEntryExits functionEntryExits;
    ASSIGN_OR_RETURN(functionEntryExits, VisitFunction(function, fn));

    functions.insert({&function, functionEntryExits});
  }

  // Add call edges to and from the root node.
  for (auto fn : functions) {
    AddCallEdges(0, fn.second);
  }

  // Add call edges to and from call sites.
  for (auto callSite : call_sites_) {
    const auto& calledFunction = functions.find(callSite.second);
    if (calledFunction == functions.end()) {
      return Status(error::Code::INVALID_ARGUMENT,
                    "Could not resolve call to function");
    }
    RETURN_IF_ERROR(AddCallEdges(callSite.first, calledFunction->second));
  }

  // Create the constants.
  for (const auto& constant : constants_) {
#ifdef PROGRAML_FUTURE_NODE_REPRESENTATION
    const string immediateText = PrintToString(*constant.first);
#else
    const string immediateText = "!IMMEDIATE";
#endif

    // Create the node for the constant.
    int immediate;
    ASSIGN_OR_RETURN(immediate, AddImmediate(immediateText));
    // Create data in-flow edges.
    for (auto destination : constant.second) {
      RETURN_IF_ERROR(
          AddDataEdge(immediate, destination.first, destination.second));
    }
  }

  return GetGraph();
}

}  // namespace ml4pl
