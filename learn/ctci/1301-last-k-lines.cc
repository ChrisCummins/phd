/*
 * Write a method to print the last K lines of an input file using C++.
 */
#include "./ctci.h"

#include <fstream>
#include <iostream>
#include <string>

//
// Optimised solution. Read characters backwards from end of file,
// stopping after k newlines.
//
void printLastKLines1(std::ifstream &file, size_t k,
                      std::ostream &out = std::cout) {
  if (!file.is_open() || !k) return;  // Sanity checks.

  // Num. of lines read, and num. of lines left to read:
  size_t linesremaining = k;
  int linesread = 0;

  // Skip to last character of file:
  file.seekg(-1, std::ios_base::end);

  while (linesremaining) {
    // Read next character.
    char ch = 0;
    file.get(ch);

    if (file.tellg() <= 1) {
      // If we have reached the end of the file, quit.
      file.seekg(0);
      linesremaining = 0;
      linesread++;
    } else {
      // If we have reached a new line, record it.
      if (ch == '\n') {
        linesremaining--;
        linesread++;
      }

      // Move back one character if there's still stuff to read.
      if (linesremaining) file.seekg(-2, std::ios_base::cur);
    }
  }

  // Print last k lines.
  std::string line;
  for (auto j = 0; j < linesread; j++) {
    getline(file, line);
    out << line << std::endl;
  }
}

///////////
// Tests //
///////////

TEST(LastKLines, tests) {
  std::string filename("1301-last-k-lines-test.txt");
  std::ifstream in;
  size_t num_lines = 0;

  for (num_lines = 0; num_lines <= 10; num_lines++) {
    std::cout << "Last " << num_lines << " lines:\n";
    in.open(filename);
    printLastKLines1(in, num_lines);
    in.close();
    std::cout << '\n';
  }
}

CTCI_MAIN();
