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
#include "deeplearning/ml4pl/seq/ir2seq.pb.h"

#include "labm8/cpp/pbutil.h"

namespace ml4pl {

// Process a graph encoder job inplace.
void ProgressStringEncoderJobInplace(StringEncoderJob* job) {
  // Create the vocabulary.
  absl::flat_hash_map<string, int> vocabulary;
  for (auto it = job->vocabulary().begin(); it != job->vocabulary().end();
       ++it) {
    vocabulary.insert({it->first, it->second});
  }

  // Create the string encoder.
  CachedStringEncoder encoder(vocabulary);

  // Encode each of the string and record the results.
  for (const auto& string : job->string()) {
    EncodedString* message = job->add_seq();
    std::vector<int> encoded = encoder.EncodeAndCache(string);
    message->mutable_encoded()->Reserve(encoded.size());
    for (int i = 0; i < encoded.size(); ++i) {
      message->mutable_encoded()->Add(encoded[i]);
    }
  }

  // Unset the input fields to minimize the size of the proto that must be
  // printed.
  job->clear_vocabulary();
  job->clear_string();
}

}  // namespace ml4pl

PBUTIL_INPLACE_PROCESS_MAIN(ml4pl::ProgressStringEncoderJobInplace,
                            ml4pl::StringEncoderJob);
