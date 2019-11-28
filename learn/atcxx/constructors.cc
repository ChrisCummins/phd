// Essential operations.

#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
#pragma GCC diagnostic ignored "-Wmissing-noreturn"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wshift-sign-overflow"
#pragma GCC diagnostic ignored "-Wundef"
#pragma GCC diagnostic ignored "-Wused-but-marked-unused"
#pragma GCC diagnostic ignored "-Wweak-vtables"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

class X {
 public:
  int* data;

  // Normal constructor:
  X(int v1, int v2) {
    data = new int;
    *data = v1 + v2;
  }

  // Explicit constructor:
  explicit X(int val) {
    data = new int;
    *data = val;
  }

  // Default constructor:
  X() {
    data = new int;
    *data = 5;
  }

  // Copy constructor:
  X(const X& x) {
    data = new int;
    *data = *x.data;
  }

  // Move constructor:
  X(X&& x) {
    data = x.data;
    x.data = nullptr;
  }

  // Copy assignment:
  X& operator=(const X& x) {
    *data = *x.data;
    return *this;
  }

  // Move assignment:
  X& operator=(X&& x) {
    if (data != x.data) {
      delete data;
      data = std::move(x.data);
    }
    return *this;
  }

  // Destructor:
  ~X() { delete data; }  // NOLINT

  // Some operator.
  X operator+(const X& x) { return X(*x.data + *data); }
};

class Y {
 public:
  int data;

  explicit Y(const int n) : data(n) { std::cout << "-> Y(const int n)\n"; }

  Y() : data(0) { std::cout << "-> Y()\n"; }

  ~Y() { std::cout << "-> ~Y()\n"; }

  Y(const Y& y) {
    data = y.data;
    std::cout << "-> Y(const Y&y)\n";
  }

  Y(Y&& y) {
    data = y.data;
    std::cout << "-> Y(Y&&y)\n";
  }

  Y& operator=(const Y& y) {
    std::cout << "-> Y& operator=(const Y& y)\n";
    data = y.data;
    return *this;
  }

  Y& operator=(Y&& y) {
    std::cout << "-> Y& operator=(Y&& y)\n";
    data = y.data;
    return *this;
  }

  Y& operator+(const Y& y) {
    std::cout << "-> Y& operator+(const Y& y)\n";
    data += y.data;
    return *this;
  }

  Y& operator+(const Y&& y) {
    std::cout << "-> Y& operator+(const Y&& y)\n";
    data += y.data;
    return *this;
  }
};

TEST(Constructors, normal) {
  X x1(10, 5);
  ASSERT_EQ(15, *x1.data);

  X x2(-3, 3);
  ASSERT_EQ(0, *x2.data);
}

TEST(Constructors, explicit) {
  X x1(10);
  ASSERT_EQ(10, *x1.data);

  X x2(0);
  ASSERT_EQ(0, *x2.data);
}

TEST(Constructors, default) { ASSERT_EQ(5, *X().data); }

TEST(Constructors, copyConstructor) {
  X x1(10);
  X x2 = x1;

  *x1.data = 15;

  ASSERT_EQ(15, *x1.data);
  ASSERT_EQ(10, *x2.data);
  // ASSERT_NEQ(x1.data, x2.data);
}

// FIXME:
//
// TEST(Constructors, moveConstructor) {
//   X x1(X(10) + X(15));
//
//   ASSERT_EQ(25, *x1.data);
//
//   X x2(10);
//
//   x2 = std::move(x1);
//
//   ASSERT_EQ(25, *x2.data);
// }
//
TEST(Constructors, copyAssignment) {
  X x1(10), x2(15);

  x1 = x2;

  ASSERT_EQ(15, *x1.data);
  ASSERT_EQ(15, *x2.data);
}
//
// TEST(Constructors, moveAssignment) {
//   X x1(10);
//
//   x1 = X(15) + X(10);
//
//   ASSERT_EQ(25, *x1.data);
// }

int main(int argc, char** argv) {
  Y y1(5);
  Y y2(10);
  Y y3(y2);
  y3 = y2;
  y2 = Y(12) + Y(15);
  y1 = std::move(y3);

  for (int i = 0; i < 3; i++) {
    Y tmp;
    y2 = tmp;
  }

  std::cout << "entering scope\n";
  { Y tmp(5); }
  std::cout << "left scope\n";

  // Run unit tests:
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
