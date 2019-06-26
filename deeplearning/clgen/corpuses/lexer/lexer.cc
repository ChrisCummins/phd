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
#include "deeplearning/clgen/corpuses/lexer/lexer.h"

namespace clgen {

bool HasPrefix(const absl::flat_hash_set<string>& strings,
               const std::string& prefix) {
  for (auto& s : strings) {
    if (s.rfind(prefix, 0) == 0) {
      return true;
    }
  }
  return false;
}

bool HasMatch(const absl::flat_hash_set<string>& strings,
              const std::string& match) {
  for (auto& s : strings) {
    if (s == match) {
      return true;
    }
  }
  return false;
}

int GetOrInsertToken(const string& token,
                     absl::flat_hash_map<string, int>* vocabulary) {
  auto encoded = vocabulary->find(token);
  if (encoded == vocabulary->end()) {
    int new_encoded = vocabulary->size();
    vocabulary->insert({token, new_encoded});
    return new_encoded;
  }
  return encoded->second;
}

std::vector<int> TokenizeInput(
    const std::string& input,
    const absl::flat_hash_set<string>& candidate_vocabulary,
    absl::flat_hash_map<string, int>* vocabulary) {
  // Given an input of length n and candidate_vocabulary of size m, this is a
  // O(n*m) algorithm that is hard to reason about. There are presumably nicer
  // and more elegant algorithms for performing this task, so consider a
  // reimplementation at some point.
  std::vector<int> tokenized;
  int i = 0;
  int j = 2;
  // Proceed through the text using two iterators to isolate a substring
  // input[i:j].
  while (i < input.size()) {
    if (j <= input.size() &&
        HasPrefix(candidate_vocabulary, input.substr(i, j - i))) {
      // input[i:j] prefixes an element in the candidate vocabulary, so advance
      // to include another character.
      ++j;
    } else {
      // input[i:j] doesn't match any prefixes in the vocabulary.
      while (j > i + 1) {
        // input[i:j] has advanced past a single character, so backtrack 'j' to
        // the last point where input[i:j] _fully_ matched a string in the
        // candidate vocabulary.
        if (HasMatch(candidate_vocabulary, input.substr(i, j - i))) {
          const int encoded =
              GetOrInsertToken(input.substr(i, j - i), vocabulary);
          tokenized.push_back(encoded);
          i = j;
          j = i + 2;
          break;
        } else {
          // Continue backtracking.
          --j;
        }
      }
      if (j == i + 1) {
        // We reached the point where input[i:j] is a single character without
        // finding a match in the vocabulary, so add the single character to the
        // vocabulary and move on.
        const int encoded = GetOrInsertToken(input.substr(i, 1), vocabulary);
        tokenized.push_back(encoded);
        i += 1;
        j += 2;
      }
    }
  }

  return tokenized;
}

void ProcessLexerJob(LexerJob* input,
                     const absl::flat_hash_set<string>& candidate_vocabulary,
                     absl::flat_hash_map<string, int>* vocabulary) {
  auto tokenized =
      TokenizeInput(input->string(), candidate_vocabulary, vocabulary);
  for (int j = 0; j < tokenized.size(); ++j) {
    input->add_token(tokenized[j]);
  }
  input->clear_string();
}

void ProcessLexerBatchJobOrDie(LexerBatchJob* proto) {
  // Initialize vocabulary.
  absl::flat_hash_map<string, int> vocabulary;
  for (auto it = proto->vocabulary().begin(); it != proto->vocabulary().end();
       ++it) {
    vocabulary.insert({it->first, it->second});
  }

  // Initialize candidate vocabulary.
  absl::flat_hash_set<string> candidate_vocabulary;
  for (int i = 0; i < proto->candidate_token_size(); ++i) {
    candidate_vocabulary.insert(proto->candidate_token(i));
  }
  proto->clear_candidate_token();

  for (int i = 0; i < proto->input_size(); ++i) {
    LexerJob* input = proto->mutable_input(i);
    ProcessLexerJob(input, candidate_vocabulary, &vocabulary);
  }

  // Write the vocabulary pairs.
  for (auto& token_pair : vocabulary) {
    proto->mutable_vocabulary()->insert({token_pair.first, token_pair.second});
  }
}

}  // namespace clgen
