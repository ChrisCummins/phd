// This file defines methods for string-ifying XLA components.
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

#include "labm8/cpp/string.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace programl {
namespace ir {
namespace xla {

using ::xla::HloInstructionProto;
using ::xla::LiteralProto;
using ::xla::ShapeProto;

string ShapeProtoToString(const ShapeProto &shape);

string HloInstructionToText(const HloInstructionProto &instruction);

string LiteralProtoToText(const LiteralProto &literal);

}  // namespace xla
}  // namespace ir
}  // namespace programl