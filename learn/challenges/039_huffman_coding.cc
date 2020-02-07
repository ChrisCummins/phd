// Implement Huffman coding.
#include "labm8/cpp/test.h"

#include <queue>
#include <unordered_map>
#include <vector>
#include "labm8/cpp/logging.h"

using std::ostream;
using std::priority_queue;
using std::unordered_map;
using std::vector;

// Time: O(n)
// Space: O(n)
unordered_map<char, size_t> BuildFrequencyMap(const string& S) {
  unordered_map<char, size_t> freq;
  for (auto c : S) {
    freq[c] += 1;
  }
  return freq;
}

TEST(HuffmanCoding, BuildFrequencyMap) {
  unordered_map<char, size_t> freq = BuildFrequencyMap("aaabcdb");

  EXPECT_EQ(freq.size(), 4);
  EXPECT_EQ(freq['a'], 3);
  EXPECT_EQ(freq['b'], 2);
  EXPECT_EQ(freq['c'], 1);
  EXPECT_EQ(freq['d'], 1);
}

class HuffmanTree {
 public:
  HuffmanTree(int freq_, HuffmanTree* left_, HuffmanTree* right_)
      : freq(freq_), c(0), left(left_), right(right_) {}
  HuffmanTree(int freq_, char c_)
      : freq(freq_), c(c_), left(nullptr), right(nullptr) {}
  HuffmanTree(int freq_, char c_, HuffmanTree* left_, HuffmanTree* right_)
      : freq(freq_), c(c_), left(left_), right(right_) {}

  int freq;
  char c;
  HuffmanTree* left;
  HuffmanTree* right;

  bool operator==(const HuffmanTree& other) const {
    return freq == other.freq && c == other.c;
  }

  friend ostream& operator<<(ostream& os, const HuffmanTree& tree);
};

ostream& operator<<(ostream& os, const HuffmanTree& tree) {
  os << '(' << tree.freq << '/' << tree.c << ' ';
  if (tree.left) {
    os << *tree.left;
  } else {
    os << "-";
  }
  os << " ";

  if (tree.right) {
    os << *tree.right;
  } else {
    os << "-";
  }
  os << ')';
  return os;
}

struct MinHeap {
  int operator()(const HuffmanTree* p1, const HuffmanTree* p2) {
    return p1->freq >= p2->freq;
  }
};

TEST(HuffmanCoding, MinHeap) {
  priority_queue<HuffmanTree*, vector<HuffmanTree*>, MinHeap> heap;
  heap.push(new HuffmanTree(10, 'a'));
  heap.push(new HuffmanTree(5, 'b'));
  heap.push(new HuffmanTree(15, 'c'));
  heap.push(new HuffmanTree(30, 'd'));

  ASSERT_EQ(heap.size(), 4);

  EXPECT_EQ(*heap.top(), HuffmanTree({5, 'b'}));
  heap.pop();
  EXPECT_EQ(*heap.top(), HuffmanTree({10, 'a'}));
  heap.pop();
  EXPECT_EQ(*heap.top(), HuffmanTree({15, 'c'}));
  heap.pop();
  EXPECT_EQ(*heap.top(), HuffmanTree({30, 'd'}));
}

HuffmanTree* BuildHuffmanTree(const string& S,
                              const unordered_map<char, size_t>& freq) {
  priority_queue<HuffmanTree*, vector<HuffmanTree*>, MinHeap> heap;
  for (auto entry : freq) {
    heap.push(new HuffmanTree(entry.second, entry.first));
  }

  while (heap.size() > 1) {
    HuffmanTree* left = new HuffmanTree(heap.top()->freq, heap.top()->c,
                                        heap.top()->left, heap.top()->right);
    heap.pop();
    HuffmanTree* right = new HuffmanTree(heap.top()->freq, heap.top()->c,
                                         heap.top()->left, heap.top()->right);
    heap.pop();

    heap.push(new HuffmanTree(left->freq + right->freq,
                              left->freq < right->freq ? left : right,
                              left->freq < right->freq ? right : left));
  }

  return heap.top();
}

TEST(HuffmanCoding, BuildHuffmanTree) {
  const string S = "aaabcdb";
  //      7
  //     / \
  //    3   4
  //       / \
  //      2   2
  //         / \
  //        1   1
  unordered_map<char, size_t> freq = BuildFrequencyMap(S);
  HuffmanTree* tree = BuildHuffmanTree(S, freq);

  EXPECT_NE(tree->left, nullptr);
  EXPECT_NE(tree->right, nullptr);

  EXPECT_EQ(tree->freq, 7);
  EXPECT_EQ(tree->c, 0);

  EXPECT_EQ(tree->left->freq, 3);
  EXPECT_EQ(tree->left->c, 'a');
  EXPECT_EQ(tree->left->left, nullptr);
  EXPECT_EQ(tree->left->right, nullptr);

  EXPECT_EQ(tree->right->freq, 4);
  EXPECT_EQ(tree->right->c, 0);

  EXPECT_EQ(tree->right->left->freq, 2);
  EXPECT_EQ(tree->right->left->c, 'b');
  EXPECT_EQ(tree->right->left->left, nullptr);
  EXPECT_EQ(tree->right->left->right, nullptr);

  EXPECT_EQ(tree->right->right->freq, 2);
  EXPECT_EQ(tree->right->right->c, 0);

  EXPECT_EQ(tree->right->right->left->freq, 1);
  EXPECT_EQ(tree->right->right->left->c, 'd');

  EXPECT_EQ(tree->right->right->right->freq, 1);
  EXPECT_EQ(tree->right->right->right->c, 'c');
}

void GetCodes(const HuffmanTree* tree, unordered_map<char, string>* codes,
              string* code, int i) {
  if (tree->left != nullptr) {
    (*code)[i] = '0';
    GetCodes(tree->left, codes, code, i + 1);
  }

  if (tree->right != nullptr) {
    (*code)[i] = '1';
    GetCodes(tree->right, codes, code, i + 1);
  }

  if (tree->left == nullptr && tree->right == nullptr) {
    codes->insert({tree->c, code->substr(0, i)});
  }
}

unordered_map<char, string> GetCodes(const HuffmanTree* tree,
                                     const unordered_map<char, size_t>& freq) {
  unordered_map<char, string> codes;
  codes.reserve(freq.size());

  // max code length is height of the tree - 1.
  // height of tree never exceeds nodes - 1.
  string code(2 * freq.size(), '-');

  GetCodes(tree, &codes, &code, 0);

  return codes;
}

TEST(HuffmanCoding, GetCodes) {
  const string S = "aaabcdb";
  //      7
  //     / \
  //    3   4
  //       / \
  //      2   2
  //         / \
  //        1   1
  unordered_map<char, size_t> freq = BuildFrequencyMap(S);
  const HuffmanTree* const tree = BuildHuffmanTree(S, freq);
  const unordered_map<char, string> codes = GetCodes(tree, freq);

  EXPECT_EQ(codes.size(), 4);
  EXPECT_EQ(codes.find('a')->second, "0");
  EXPECT_EQ(codes.find('b')->second, "10");
  EXPECT_EQ(codes.find('c')->second, "111");
  EXPECT_EQ(codes.find('d')->second, "110");
}

string Encode(const string& S, const unordered_map<char, string>& codes) {
  string encoded;
  encoded.reserve(S.size());

  for (auto c : S) {
    auto it = codes.find(c);
    CHECK(it != codes.end()) << "Failed to lookup char " << c;
    encoded += it->second;
  }

  return encoded;
}

TEST(HuffmanCoding, Encode) {
  const string S = "aaabcdb";
  unordered_map<char, size_t> freq = BuildFrequencyMap(S);
  const HuffmanTree* const tree = BuildHuffmanTree(S, freq);
  const unordered_map<char, string> codes = GetCodes(tree, freq);

  auto encoded = Encode(S, codes);

  EXPECT_EQ(encoded, "0001011111010");
}

TEST_MAIN();
