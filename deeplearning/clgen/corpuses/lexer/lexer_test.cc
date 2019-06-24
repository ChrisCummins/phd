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

#include "deeplearning/clgen/proto/internal.pb.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

#include "phd/string.h"
#include "phd/test.h"

namespace clgen {
namespace {

TEST(HasPrefix, EmptyVocabDoesNotMatchString) {
  absl::flat_hash_set<string> vocab;
  ASSERT_FALSE(HasPrefix(vocab, "a"));
}

TEST(HasPrefix, EntireStringMatchesSingleElementSet) {
  absl::flat_hash_set<string> vocab;
  vocab.insert("a");
  ASSERT_TRUE(HasPrefix(vocab, "a"));
}

TEST(HasPrefix, StringPrefixMatchesSingleElementSet) {
  absl::flat_hash_set<string> vocab;
  vocab.insert("abc");
  ASSERT_TRUE(HasPrefix(vocab, "ab"));
}

TEST(HasPrefix, StringPrefixDoesNotMatchSingleElementSet) {
  absl::flat_hash_set<string> vocab;
  vocab.insert("abc");
  ASSERT_FALSE(HasPrefix(vocab, "bc"));
}

TEST(HasPrefix, StringPrefixMatchesMultiElementSet) {
  absl::flat_hash_set<string> vocab;
  vocab.insert("abc");
  vocab.insert("bcd");
  ASSERT_TRUE(HasPrefix(vocab, "ab"));
}

TEST(HasPrefix, StringPrefixDoesNotMatchMultiElementSet) {
  absl::flat_hash_set<string> vocab;
  vocab.insert("abc");
  vocab.insert("bcd");
  ASSERT_FALSE(HasPrefix(vocab, "c"));
}

absl::flat_hash_map<int, string> GetReverseVocab(
    const absl::flat_hash_map<string, int>& vocab) {
  absl::flat_hash_map<int, string> rvocab;
  for (auto val : vocab) {
    rvocab.insert({val.second, val.first});
  }
  return rvocab;
}

TEST(TokenizeInput, NoVocabMatches) {
  absl::flat_hash_set<string> candidate_vocab;
  candidate_vocab.insert("d");
  absl::flat_hash_map<string, int> vocab;
  auto tokenized = TokenizeInput("abca", candidate_vocab, &vocab);
  auto rvocab = GetReverseVocab(vocab);

  ASSERT_EQ(3, vocab.size());
  ASSERT_NE(vocab.end(), vocab.find("a"));
  ASSERT_NE(vocab.end(), vocab.find("b"));
  ASSERT_NE(vocab.end(), vocab.find("c"));

  ASSERT_EQ(4, tokenized.size());
  ASSERT_EQ("a", rvocab.find(tokenized[0])->second);
  ASSERT_EQ("b", rvocab.find(tokenized[1])->second);
  ASSERT_EQ("c", rvocab.find(tokenized[2])->second);
  ASSERT_EQ("a", rvocab.find(tokenized[3])->second);
}

TEST(TokenizeInput, SingleCharInputAndVocab) {
  absl::flat_hash_set<string> candidate_vocab;
  candidate_vocab.insert("ab");
  absl::flat_hash_map<string, int> vocab;
  auto tokenized = TokenizeInput("aaba", candidate_vocab, &vocab);
  auto rvocab = GetReverseVocab(vocab);

  ASSERT_EQ(2, vocab.size());
  ASSERT_NE(vocab.end(), vocab.find("ab"));
  ASSERT_NE(vocab.end(), vocab.find("a"));

  ASSERT_EQ(3, tokenized.size());
  ASSERT_EQ("a", rvocab.find(tokenized[0])->second);
  ASSERT_EQ("ab", rvocab.find(tokenized[1])->second);
  ASSERT_EQ("a", rvocab.find(tokenized[2])->second);
}

TEST(TokenizeInput, LongestVocabElementMatches) {
  absl::flat_hash_set<string> candidate_vocab;
  candidate_vocab.insert("ab");
  candidate_vocab.insert("abcd");
  absl::flat_hash_map<string, int> vocab;
  auto tokenized = TokenizeInput("abcd", candidate_vocab, &vocab);
  auto rvocab = GetReverseVocab(vocab);

  ASSERT_EQ(1, vocab.size());
  ASSERT_NE(vocab.end(), vocab.find("abcd"));

  ASSERT_EQ(1, tokenized.size());
  ASSERT_EQ("abcd", rvocab.find(tokenized[0])->second);
}

TEST(TokenizeInput, GreedyVocabMatches) {
  absl::flat_hash_set<string> candidate_vocab;
  candidate_vocab.insert("abc");
  candidate_vocab.insert("cdef");
  absl::flat_hash_map<string, int> vocab;
  auto tokenized = TokenizeInput("abcdef", candidate_vocab, &vocab);
  auto rvocab = GetReverseVocab(vocab);

  ASSERT_EQ(4, vocab.size());
  ASSERT_NE(vocab.end(), vocab.find("abc"));
  ASSERT_NE(vocab.end(), vocab.find("d"));
  ASSERT_NE(vocab.end(), vocab.find("e"));
  ASSERT_NE(vocab.end(), vocab.find("f"));

  ASSERT_EQ(4, tokenized.size());
  ASSERT_EQ("abc", rvocab.find(tokenized[0])->second);
  ASSERT_EQ("d", rvocab.find(tokenized[1])->second);
  ASSERT_EQ("e", rvocab.find(tokenized[2])->second);
  ASSERT_EQ("f", rvocab.find(tokenized[3])->second);
}

TEST(TokenizeInput, SmallCProgram) {
  absl::flat_hash_set<string> candidate_vocab;
  candidate_vocab.insert("int");
  candidate_vocab.insert("void");
  candidate_vocab.insert("main");
  candidate_vocab.insert("float");  // unused
  candidate_vocab.insert("return");
  absl::flat_hash_map<string, int> vocab;
  auto tokenized =
      TokenizeInput("int void main() { return 0; }", candidate_vocab, &vocab);
  auto rvocab = GetReverseVocab(vocab);

  ASSERT_EQ(11, vocab.size());
  ASSERT_EQ(vocab.end(), vocab.find("float"));  // unused
  ASSERT_NE(vocab.end(), vocab.find("int"));
  ASSERT_NE(vocab.end(), vocab.find(" "));
  ASSERT_NE(vocab.end(), vocab.find("main"));
  ASSERT_NE(vocab.end(), vocab.find("("));
  ASSERT_NE(vocab.end(), vocab.find(")"));
  ASSERT_NE(vocab.end(), vocab.find("{"));
  ASSERT_NE(vocab.end(), vocab.find("return"));
  ASSERT_NE(vocab.end(), vocab.find("0"));
  ASSERT_NE(vocab.end(), vocab.find(";"));
  ASSERT_NE(vocab.end(), vocab.find("}"));

  ASSERT_EQ(16, tokenized.size());
  ASSERT_EQ("int", rvocab.find(tokenized[0])->second);
  ASSERT_EQ(" ", rvocab.find(tokenized[1])->second);
  ASSERT_EQ("void", rvocab.find(tokenized[2])->second);
  ASSERT_EQ(" ", rvocab.find(tokenized[3])->second);
  ASSERT_EQ("main", rvocab.find(tokenized[4])->second);
  ASSERT_EQ("(", rvocab.find(tokenized[5])->second);
  ASSERT_EQ(")", rvocab.find(tokenized[6])->second);
  ASSERT_EQ(" ", rvocab.find(tokenized[7])->second);
  ASSERT_EQ("{", rvocab.find(tokenized[8])->second);
  ASSERT_EQ(" ", rvocab.find(tokenized[9])->second);
  ASSERT_EQ("return", rvocab.find(tokenized[10])->second);
  ASSERT_EQ(" ", rvocab.find(tokenized[11])->second);
  ASSERT_EQ("0", rvocab.find(tokenized[12])->second);
  ASSERT_EQ(";", rvocab.find(tokenized[13])->second);
  ASSERT_EQ(" ", rvocab.find(tokenized[14])->second);
  ASSERT_EQ("}", rvocab.find(tokenized[15])->second);
}

TEST(ProcessLexerJob, SimpleStringInput) {
  LexerJob job;
  job.set_string("abc");
  absl::flat_hash_set<string> candidate_vocabulary;
  candidate_vocabulary.insert("bc");
  absl::flat_hash_map<string, int> vocabulary;

  ProcessLexerJob(&job, candidate_vocabulary, &vocabulary);
  ASSERT_EQ(2, vocabulary.size());
  ASSERT_EQ(0, vocabulary.find("a")->second);
  ASSERT_EQ(1, vocabulary.find("bc")->second);

  ASSERT_EQ(2, job.token_size());
  ASSERT_EQ(0, job.token(0));
  ASSERT_EQ(1, job.token(1));
}

}  // namespace
}  // namespace clgen

TEST_MAIN();
