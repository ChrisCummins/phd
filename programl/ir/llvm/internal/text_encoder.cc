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
#include "programl/ir/llvm/internal/text_encoder.h"

#include <sstream>
#include <utility>

using std::make_pair;

namespace programl {
namespace ir {
namespace llvm {
namespace internal {

string TextEncoder::GetInstructionLhs(const ::llvm::Instruction* instruction) {
  return Encode(instruction).first;
}

string TextEncoder::GetInstructionRhs(const ::llvm::Instruction* instruction) {
  return Encode(instruction).second;
}

void TextEncoder::Clear() { texts_.clear(); }

pair<string, string> TextEncoder::Encode(
    const ::llvm::Instruction* instruction) {
  auto it = texts_.find(instruction);
  if (it != texts_.end()) {
    return it->second;
  }

  const string instructionString = PrintToString(*instruction);
  const size_t snipAt = instructionString.find(" = ");

  // An instruction without a LHS.
  if (snipAt == string::npos) {
    pair<string, string> encoded = make_pair("", instructionString);
    texts_.insert({instruction, encoded});
    return encoded;
  }

  const string identifier = instructionString.substr(0, snipAt);
  const string type = PrintToString(*instruction->getType());

  std::stringstream instructionName;
  instructionName << type << ' ' << identifier;

  string lhs = instructionName.str();
  string rhs = instructionString.substr(snipAt + 3);
  pair<string, string> encoded = make_pair(lhs, rhs);

  texts_.insert({instruction, encoded});
  return encoded;
}

}  // namespace internal
}  // namespace llvm
}  // namespace ir
}  // namespace programl
