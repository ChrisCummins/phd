// You are given the following information, but you may prefer to do some
// research for yourself.
//
// * 1 Jan 1900 was a Monday.
// * Thirty days has September, April, June and November.
//   All the rest have thirty-one,
//   Saving February alone, Which has twenty-eight, rain or shine.
//   And on leap years, twenty-nine.
// * A leap year occurs on any year evenly divisible by 4, but not on a
//   century unless it is divisible by 400.
//
// How many Sundays fell on the first of the month during the twentieth
// century (1 Jan 1901 to 31 Dec 2000)?

#include <cassert>
#include <iostream>

int DaysInMonth(const int& y, const int& m) {
  int dim = 31;
  if (m == 9 || m == 4 || m == 6 || m == 11) {
    dim = 30;
  } else if (m == 2) {
    dim = y % 4 == 0 ? y % 100 == 0 ? 29 : y % 400 == 0 ? 29 : 28 : 28;
  }
  return dim;
}

void AdvanceOneWeek(int* const y, int* const m, int* const d) {
  int dim = DaysInMonth(*y, *m);

  *d += 7;
  if (*d > dim) {
    *d = *d % dim;
    *m = *m + 1;
    if (*m > 12) {
      *m = 1;
      *y = *y + 1;
    }
  }
}

int NumSundays() {
  int c = 0;

  int y = 1900;
  int m = 1;
  int d = 7;

  // Advance to first sunday in 1901.
  while (y < 1901) {
    AdvanceOneWeek(&y, &m, &d);
  }

  // Count the sundays between [1901,2000].
  while (y < 2001) {
    AdvanceOneWeek(&y, &m, &d);
    if (d == 1) {
      ++c;
    }
  }

  return c;
}

int main(int argc, char** argv) {
  int numSundays = NumSundays();
  std::cout << numSundays << std::endl;
  assert(numSundays == 171);
  return 0;
}
