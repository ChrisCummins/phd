#pragma once

#include <vector>

#include "deeplearning/ml4pl/graphs/programl.pb.h"
#include "deeplearning/ml4pl/seq/cached_string_encoder.h"
#include "deeplearning/ml4pl/seq/graph2seq.pb.h"

namespace ml4pl {

class GraphEncoder {
 public:
  GraphEncoder(CachedStringEncoder& string_encoder)
      : string_encoder_(string_encoder) {}

  ProgramGraphSeq Encode(const ProgramGraph& graph);

 private:
  CachedStringEncoder string_encoder_;
};

}  // namespace ml4pl
