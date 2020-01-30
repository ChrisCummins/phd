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
#include "deeplearning/ml4pl/seq/cached_string_encoder.h"
#include "deeplearning/ml4pl/seq/graph2seq.pb.h"
#include "deeplearning/ml4pl/seq/graph_encoder.h"

#include "labm8/cpp/pbutil.h"

namespace ml4pl {

// Process a graph encoder job inplace.
void ProgressGraphEncoderJobInplace(GraphEncoderJob* job) {
  // Create the vocabulary.
  absl::flat_hash_map<string, int> vocabulary;
  for (auto it = job->vocabulary().begin(); it != job->vocabulary().end();
       ++it) {
    vocabulary.insert({it->first, it->second});
  }

  // Create the string and graph encoders.
  CachedStringEncoder string_encoder(vocabulary);
  GraphEncoder encoder(string_encoder);

  // Encode each of the graphs and record the results.
  for (const auto& graph : job->graph()) {
    *job->add_seq() = encoder.Encode(graph);
  }

  // Unset the input fields to minimize the size of the proto that must be
  // printed.
  job->clear_vocabulary();
  job->clear_graph();
}

}  // namespace ml4pl

PBUTIL_INPLACE_PROCESS_MAIN(ml4pl::ProgressGraphEncoderJobInplace,
                            ml4pl::GraphEncoderJob);
