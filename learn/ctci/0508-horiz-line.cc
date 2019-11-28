/*
 * A monochrome screen is stored as a single array of bytes, allowing
 * eight consecutive pixels to be stored in one byte. The screen has
 * width w, where w is divisble by 8 (that is, no byte will be split
 * across rows). The height of the screen, of course, can be derived
 * from the length of the array and the width. Implement a function
 * drawHorizontalLine(byte[] screen, int width, int x1, int x2, int y)
 * which draws a horizontal line from (x1, y) to (x2, y).
 */
#include "./ctci.h"

void drawHorizontalLine(unsigned char *screen, unsigned int width,
                        unsigned int x1, unsigned int x2, unsigned int y) {
  // const auto start_idx = x1 / 8;
  // const auto start_offset = x1 % 8;

  const auto last_idx = x2 / 8;
  // const auto last_offset = x2 % 8;

  if (x2 - x1 < 8) {
    // TODO: single-byte mask.
  } else {
    // TODO: mask first and last byte.
    const auto start_inner = x1 / 8 + 1;
    for (auto i = start_inner; i < last_idx; i++) screen[i] = 0xff;
  }
}

///////////
// Tests //
///////////
TEST(drawHorizontalLine, drawHorizontalLine) {}

CTCI_MAIN();
