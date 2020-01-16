/*
 * Design an algorithm to figure out if someone has won a game of
 * tic-tac-toe.
 */
#include "./ctci.h"

#include <array>

static unsigned int seed = 0xCEC;

using Board = std::array<std::array<char, 3>, 3>;

inline void zero_board(Board *b) {
  for (auto &row : *b) {
    for (auto &cell : row) {
      cell = '-';
    }
  }
}

//
// There are eight possible solutions. Test for each.
//
bool player_won(const Board &b, const char player) {
  // Rows:
  if (b[0][0] == player && b[1][0] == player && b[2][0] == player)
    return true;
  else if (b[0][1] == player && b[1][1] == player && b[2][1] == player)
    return true;
  else if (b[0][2] == player && b[1][2] == player && b[2][2] == player)
    return true;

  // Columns:
  else if (b[0][0] == player && b[0][1] == player && b[0][2] == player)
    return true;
  else if (b[1][0] == player && b[1][1] == player && b[1][2] == player)
    return true;
  else if (b[2][0] == player && b[2][1] == player && b[2][2] == player)
    return true;

  // Diagonals:
  else if (b[0][0] == player && b[1][1] == player && b[2][2] == player)
    return true;
  else if (b[2][0] == player && b[1][1] == player && b[0][2] == player)
    return true;

  else
    return false;
}

///////////
// Tests //
///////////

TEST(TicTacToe, EmptyBoard) {
  Board b;
  zero_board(&b);

  ASSERT_EQ(false, player_won(b, 'x'));
  ASSERT_EQ(false, player_won(b, 'y'));

  //  - - -
  //  - - -
  //  - - -

  ASSERT_EQ(false, player_won(b, 'x'));
  ASSERT_EQ(false, player_won(b, 'y'));
}

TEST(TicTacToe, VerticalColumnWin) {
  Board b;
  zero_board(&b);

  ASSERT_EQ(false, player_won(b, 'x'));
  ASSERT_EQ(false, player_won(b, 'y'));

  //  X Y -
  //  X - Y
  //  X - Y
  b[0][0] = 'x';
  b[1][0] = 'x';
  b[2][0] = 'x';
  b[0][1] = 'y';
  b[1][2] = 'y';
  b[2][2] = 'y';

  ASSERT_EQ(tru, player_won(b, 'x'));
  ASSERT_EQ(false, player_won(b, 'y'));
}

////////////////
// Benchmarks //
////////////////

void BM_player_won(benchmark::State &state) {
  Board b;

  while (state.KeepRunning()) {
    for (size_t x = 0; x < 3; x++)
      for (size_t y = 0; y < 3; y++)
        b[x][y] = static_cast<char>(rand_r(&seed) % 3);

    player_won(b, 0);
    player_won(b, 1);
    player_won(b, 2);
    benchmark::DoNotOptimize(b);
  }
}
BENCHMARK(BM_player_won);

CTCI_MAIN();
