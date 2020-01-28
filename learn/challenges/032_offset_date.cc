// Write a function to add a number of days to a date.

#include "labm8/cpp/test.h"

struct date {
  int year;
  int month;  // [0,11]
  int day;    // [0,30]
};

// Time: O(1)
// Space: O(1)
int DaysInMonth(const date& d) {
  if (d.month == 1) {
    if ((d.year % 4 && d.year % 100) || d.year % 400) {
      return 29;  // leap year
    }
  }

  if (d.month == 3 || d.month == 5 || d.month == 8 || d.month == 10) {
    return 30;
  }

  return 31;
}

// Time: O(n)
// Space: O(1)
void OffsetDate(date* d, int n) {
  if (n < 0) {
    return;
  }

  d->day += n;

  int dim = DaysInMonth(*d);
  while (d->day >= dim) {
    d->day -= dim;
    d->month = d->month + 1;
    if (d->month >= 12) {
      d->year = d->year + 1;
      d->month -= 12;
    }
    dim = DaysInMonth(*d);
  }
}

TEST(OffsetDate, Zero) {
  date d{2000, 0, 0};
  OffsetDate(&d, 0);
  EXPECT_EQ(d.year, 2000);
  EXPECT_EQ(d.month, 0);
  EXPECT_EQ(d.day, 0);
}

TEST(OffsetDate, One) {
  date d{2000, 0, 0};
  OffsetDate(&d, 1);
  EXPECT_EQ(d.year, 2000);
  EXPECT_EQ(d.month, 0);
  EXPECT_EQ(d.day, 1);
}

TEST(OffsetDate, OneHundred) {
  date d{2020, 0, 0};
  OffsetDate(&d, 100);
  EXPECT_EQ(d.year, 2020);
  EXPECT_EQ(d.month, 3);
  EXPECT_EQ(d.day, 9);
}

TEST(OffsetDate, FourHundred) {
  date d{2020, 0, 0};
  OffsetDate(&d, 400);
  EXPECT_EQ(d.year, 2021);
  EXPECT_EQ(d.month, 1);
  EXPECT_EQ(d.day, 3);
}

TEST(OffsetDate, OneThousand) {
  date d{2020, 0, 0};
  OffsetDate(&d, 1000);
  EXPECT_EQ(d.year, 2022);
  EXPECT_EQ(d.month, 8);
  EXPECT_EQ(d.day, 24);
}

TEST_MAIN();
