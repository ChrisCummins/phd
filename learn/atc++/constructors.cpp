// Essential operations.

#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

class X {
 public:
  int *data;

  // Normal constructor:
  X(int v1, int v2) { data = new int; *data = v1 + v2; }

  // Explicit constructor:
  explicit X(int val) { data = new int; *data = val; }

  // Default constructor:
  X() { data = new int; *data = 5; }

  // Copy constructor:
  X(const X &x) { data = new int; *data = *x.data; }

  // Move constructor:
  X(X &&x) { data = x.data; x.data = nullptr; }

  // Copy assignment:
  X &operator=(const X &x) { *data = *x.data; return *this; }

  // Move assignment:
  X &operator=(X &&x) { data = x.data; x.data = nullptr; return *this; }

  // Destructor:
  ~X() { delete data; }

  // Some operator.
  X operator+(const X &x) { return X(*x.data + *data); }
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

TEST(Constructors, default) {
  ASSERT_EQ(5, *X().data);
}

TEST(Constructors, copyConstructor) {
  X x1(10);
  X x2 = x1;

  *x1.data = 15;

  ASSERT_EQ(15, *x1.data);
  ASSERT_EQ(10, *x2.data);
  // ASSERT_NEQ(x1.data, x2.data);
}

TEST(Constructors, moveConstructor) {
  X x1(X(10) + X(15));

  ASSERT_EQ(25, *x1.data);

  X x2(10);

  x2 = std::move(x1);

  ASSERT_EQ(25, *x2.data);
}

TEST(Constructors, copyAssignment) {
  X x1(10), x2(15);

  x1 = x2;

  ASSERT_EQ(15, *x1.data);
  ASSERT_EQ(15, *x2.data);
}

TEST(Constructors, moveAssignment) {
  X x1(10);

  x1 = X(15) + X(10);

  ASSERT_EQ(25, *x1.data);
}

int main(int argc, char **argv) {
  // Run unit tests:
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
