/*
 * Write a method to print the last K lines of an input file using C++.
 */
#include <fstream>
#include <iostream>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop


/*
 * Read backwards from end of file.
 */
void printLastKLines1(std::ifstream &file, size_t k,
                      std::ostream &out = std::cout) {
  // Sanity checks.
  if (!file.is_open() || !k)
    return;

  // Keep track of lines left to read.
  int linesremaining = k;
  // Keep track of lines read (where # of lines in file < k).
  int readlines = 0;

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
      readlines++;
    } else {
      // If we have reached a new line, take note.
      if (ch == '\n') {
        linesremaining--;
        readlines++;
      }

      // Move back one character if there's still stuff to read.
      if (linesremaining)
        file.seekg(-2, std::ios_base::cur);
    }
  }

  // Print last k lines.
  std::string line;
  for (auto j = 0; j < readlines; j++) {
    getline(file, line);
    out << line << std::endl;
  }
}


// Unit tests

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


int main(int argc, char **argv) {
  // Run unit tests:
  testing::InitGoogleTest(&argc, argv);
  const auto ret = RUN_ALL_TESTS();

  // Run benchmarks:
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return ret;
}
