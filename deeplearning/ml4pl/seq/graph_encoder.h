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

#include "deeplearning/ml4pl/graphs/programl.pb.h"
#include "deeplearning/ml4pl/seq/cached_string_encoder.h"
#include "deeplearning/ml4pl/seq/graph2seq.pb.h"

namespace ml4pl {

class GraphEncoder {
 public:
  GraphEncoder(const CachedStringEncoder& string_encoder)
      : string_encoder_(string_encoder) {}

  ProgramGraphSeq Encode(const ProgramGraph& graph);

 private:
  const CachedStringEncoder string_encoder_;
};

}  // namespace ml4pl
