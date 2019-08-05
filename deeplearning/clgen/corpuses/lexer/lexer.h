// Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
//
// clgen is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// clgen is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with clgen.  If not, see <https://www.gnu.org/licenses/>.
#pragma once

#include <vector>

#include "deeplearning/clgen/proto/internal.pb.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

#include "labm8/cpp/string.h"

namespace clgen {

// Determine if any of a set of strings starts with prefix.
// This assumes that strings set and prefix are not empty.
bool HasPrefix(const absl::flat_hash_set<string>& strings,
               const std::string& prefix);

// Determine if any of a s et of strings matches a string.
// This assumes that strings set and match are note empty.
bool HasMatch(const absl::flat_hash_set<string>& strings,
              const std::string& match);

// Tokenize a string into a list of tokens, where candidate_vocabulary is the
// set of multi-character tokens, and vocabulary is a dictionary of mappings
// from token to indices array.
std::vector<int> TokenizeInput(
    const std::string& input,
    const absl::flat_hash_set<string>& candidate_vocabulary,
    absl::flat_hash_map<string, int>* vocabulary);

// Process a single LexerJob inplace.
void ProcessLexerJob(LexerJob* input,
                     const absl::flat_hash_set<string>& candidate_vocabulary,
                     absl::flat_hash_map<string, int>* vocabulary);

// Process a LexerBatchJob. Any errors will lead to fatal program crash.
void ProcessLexerBatchJobOrDie(LexerBatchJob* proto);

}  // namespace clgen
