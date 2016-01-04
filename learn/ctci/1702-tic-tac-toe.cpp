/*
 * Design an algorithm to figure out if someone has won a game of
 * tic-tac-toe.
 */
#include <array>
#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

using Board = std::array<std::array<char, 3>, 3>;

void zeroBoard(Board &b) {
  for (auto &row : b)
    for (auto &cell : row)
      cell = '-';
}

void printBoard(Board &b) {
  std::cout << "Board:\n";
  for (auto &row : b) {
    for (auto &cell : row)
      std::cout << cell << " ";
    std::cout << std::endl;
  }
}

bool playerWon1(Board &b, char c) {
  /*
   * There are 8 possible winning combinations. Test for each.
   */

  // Columns:
  if (b[0][0] == c && b[1][0] == c && b[2][0] == c)
    return true;
  else if (b[0][1] == c && b[1][1] == c && b[2][1] == c)
    return true;
  else if (b[0][2] == c && b[1][2] == c && b[2][2] == c)
    return true;

  // Rows:
  else if (b[0][0] == c && b[0][1] == c && b[0][2] == c)
    return true;
  else if (b[1][0] == c && b[1][1] == c && b[1][2] == c)
    return true;
  else if (b[2][0] == c && b[2][1] == c && b[2][2] == c)
    return true;

  // Diagonals:
  else if (b[0][0] == c && b[1][1] == c && b[2][2] == c)
    return true;
  else if (b[2][0] == c && b[1][1] == c && b[0][2] == c)
    return true;

  else
    return false;
}

// Unit tests

TEST(TicTacToe, playerWon1) {
  Board b;
  zeroBoard(b);

  ASSERT_EQ(false, playerWon1(b, 'x'));
  ASSERT_EQ(false, playerWon1(b, 'y'));

  b[0][0] = 'x';
  b[1][0] = 'x';
  b[2][0] = 'x';
  b[0][1] = 'y';
  b[1][2] = 'y';
  b[2][2] = 'y';

  ASSERT_EQ(true,  playerWon1(b, 'x'));
  ASSERT_EQ(false, playerWon1(b, 'y'));
}

// Benchmarks

void BM_baseline(benchmark::State& state) {
  Board b;

  while (state.KeepRunning()) {
    for (size_t x = 0; x < 3; x++)
      for (size_t y = 0; y < 3; y++)
        b[x][y] = static_cast<char>(arc4random() % 3);

    benchmark::DoNotOptimize(b);
  }
}
BENCHMARK(BM_baseline);

void BM_playerWon1(benchmark::State& state) {
  Board b;

  while (state.KeepRunning()) {
    for (size_t x = 0; x < 3; x++)
      for (size_t y = 0; y < 3; y++)
        b[x][y] = static_cast<char>(arc4random() % 3);

    playerWon1(b, 0);
    playerWon1(b, 1);
    playerWon1(b, 2);
    benchmark::DoNotOptimize(b);
  }
}
BENCHMARK(BM_playerWon1);


int main(int argc, char **argv) {
  // Run unit tests:
  testing::InitGoogleTest(&argc, argv);
  const auto ret = RUN_ALL_TESTS();

  // Run benchmarks:
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return ret;
}
