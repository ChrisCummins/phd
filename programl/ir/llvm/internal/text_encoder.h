// This file defines utilities for encoding LLVM objects to text.
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

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "labm8/cpp/string.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

using std::pair;

namespace programl {
namespace ir {
namespace llvm {
namespace internal {

// Produce the textual representation of an LLVM value.
// This accepts instances of ::llvm::Instruction or ::llvm::Value.
template <typename T>
string PrintToString(const T &value) {
  std::string str;
  ::llvm::raw_string_ostream rso(str);
  value.print(rso);
  // Trim any leading indentation whitespace.
  labm8::TrimLeft(str);
  return str;
}

class TextEncoder {
 public:
  // Generate the split "left"- and "right"-hand side components of an
  // instruction.
  //
  //     E.g.      "%5 = add nsw i32 %3, %4"
  //          LHS: "int64* %5"
  //          RHS: "add nsw i32 %3, %4".
  //
  // Calling GetInstructionLhs() on an instruction with no LHS,
  // e.g. "ret i32 %13", returns an empty string.
  //
  // LLVM doesn't require "names" for instructions since it is in SSA form, so
  // these methods generates one by printing the instruction to a string (to
  // generate identifiers), then splitting the LHS identifier name and
  // concatenating it with the type.
  //
  // See: https://lists.llvm.org/pipermail/llvm-dev/2010-April/030726.html
  string GetInstructionLhs(const ::llvm::Instruction *instruction);

  string GetInstructionRhs(const ::llvm::Instruction *instruction);

  // Clear the encoded string cache.
  void Clear();

 private:
  pair<string, string> Encode(const ::llvm::Instruction *instruction);

  absl::flat_hash_map<const ::llvm::Instruction *, pair<string, string>> texts_;
};

}  // namespace internal
}  // namespace llvm
}  // namespace ir
}  // namespace programl
