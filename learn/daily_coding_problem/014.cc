// This problem was asked by Google.
//
// Suppose we represent our file system by a string in the following manner:
//
// The string "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext" represents:
//
// dir
//     subdir1
//     subdir2
//         file.ext
// The directory dir contains an empty sub-directory subdir1 and a sub-directory
// subdir2 containing a file file.ext.
//
// The string
// "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"
// represents:
//
// dir
//     subdir1
//         file1.ext
//         subsubdir1
//     subdir2
//         subsubdir2
//             file2.ext
// The directory dir contains two sub-directories subdir1 and subdir2. subdir1
// contains a file file1.ext and an empty second-level sub-directory subsubdir1.
// subdir2 contains a second-level sub-directory subsubdir2 containing a file
// file2.ext.
//
// We are interested in finding the longest (number of characters) absolute path
// to a file within our file system. For example, in the second example above,
// the longest absolute path is "dir/subdir2/subsubdir2/file2.ext", and its
// length is 32 (not including the double quotes).
//
// Given a string representing the file system in the above format, return the
// length of the longest absolute path to a file in the abstracted file system.
// If there is no file in the system, return 0.
//
// Note:
//
// The name of a file contains at least a period and an extension.
//
// The name of a directory or sub-directory will not contain a period.
#include <algorithm>
#include <stack>
#include <string>
#include "labm8/cpp/test.h"

using std::max;
using std::stack;
using std::string;

int F(const string& s) {
  stack<int> dirs;
  int j = 0;
  bool isf = false;
  int maxl = 0;
  int l = 0;
  size_t lvl = 0;

  for (int i = 0; i < static_cast<int>(s.size()); ++i) {
    if (s[i] == '\n') {
      if (isf) {
        maxl = max(maxl, l + i - j);
      } else {
        dirs.push(i - j);
        l += i - j;
      }
      isf = false;
      lvl = 0;
    } else if (s[i] == '\t') {
      lvl += 1;
      j = i + 1;
    } else {
      if (i == j) {
        while (dirs.size() > lvl) {
          l -= dirs.top();
          dirs.pop();
        }
      }
      if (s[i] == '.') {
        isf = true;
      }
    }
  }

  if (isf) {
    std::cout << "j = " << j << ", s.size() = " << s.size() << ", l = " << l
              << std::endl;
    maxl = max(maxl, l + static_cast<int>(s.size()) - j);
  }

  return maxl;
}

TEST(FileSystemMaxLen, EmptyString) { EXPECT_EQ(F(""), 0); }

TEST(FileSystemMaxLen, OnlyDir) { EXPECT_EQ(F("dir"), 0); }

TEST(FileSystemMaxLen, OnlyFile) { EXPECT_EQ(F("foo.txt"), 7); }

TEST(FileSystemMaxLen, OneFileInDir) { EXPECT_EQ(F("dir\n\tfoo.txt"), 10); }

TEST(FileSystemMaxLen, DeepThenShallowDir) {
  EXPECT_EQ(F("dir\n\tfoo\n\t\tfile.txt\n\tbar\n\t\ta.txt"), 14);
}

TEST(FileSystemMaxLen, ExampleInputA) {
  EXPECT_EQ(F("dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"), 18);
}

TEST(FileSystemMaxLen, ExampleInputB) {
  EXPECT_EQ(F("dir\n"
              "\tsubdir1"
              "\n\t\tfile1.ext"
              "\n\t\tsubsubdir1"
              "\n\tsubdir2"
              "\n\t\tsubsubdir2"
              "\n\t\t\tfile2.ext"),
            29);
}

TEST_MAIN();
