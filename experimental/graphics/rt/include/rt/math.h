/* -*-c++-*-
 *
 * Copyright (C) 2015, 2016 Chris Cummins.
 *
 * This file is part of rt.
 *
 * rt is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at
 * your option) any later version.
 *
 * rt is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
 * License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with rt.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef RT_MATH_H_
#define RT_MATH_H_

#include <cmath>

namespace rt {

/*
 * Maths. can't live with it, can't live without it. For ray tracing,
 * we're interested in vector and matrix manipulation. Both vector and
 * matrix types are immutable, and entirely inline. This means there's
 * no implementation file.
 */

// Changing between different floating point sizes for scalar values
// will affect the system's performance.
using Scalar = double;

// The "rounding error" to accomodate for when approximate infinite
// precision real numbers.
static const Scalar ScalarPrecision = 1e-6;

namespace radians {
// Conversion from radians to degrees.
auto inline toDegrees(const Scalar radians) { return radians * M_PI / 180.0; }
}  // namespace radians

namespace deg {
// Trigonometric functions accepting theta angles in degrees
// rather than radians:

// Ignore recursion warning from deg::sin() clashing with cmath's
// sin():
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winfinite-recursion"
Scalar inline sin(const Scalar theta) { return sin(radians::toDegrees(theta)); }

Scalar inline cos(const Scalar theta) { return cos(radians::toDegrees(theta)); }
#pragma GCC diagnostic pop

}  // namespace deg

// Clamp a Scalar value to within the range [0,1].
Scalar inline clamp(const Scalar x) {
  if (x > 1)
    return 1;
  else if (x < 0)
    return 0;
  else
    return x;
}

// A vector consists of three coordinates and a translation
// scalar. Vectors are immutable.
class Vector {
 public:
  const Scalar x;
  const Scalar y;
  const Scalar z;
  const Scalar w;

  // Contructor: V = (x,y,z,w)
  inline Vector(const Scalar _x, const Scalar _y, const Scalar _z = 0,
                const Scalar _w = 0)
      : x(_x), y(_y), z(_z), w(_w) {}

  // Addition: A' = A + B
  auto inline operator+(const Vector &b) const {
    return Vector(x + b.x, y + b.y, z + b.z);
  }

  // Subtraction: A' = A - B
  auto inline operator-(const Vector &b) const {
    return Vector(x - b.x, y - b.y, z - b.z);
  }

  // Multiplication: A' = aA
  auto inline operator*(const Scalar a) const {
    return Vector(a * x, a * y, a * z);
  }

  // Division: A' = A / a
  auto inline operator/(const Scalar a) const {
    return Vector(x / a, y / a, z / a);
  }

  // Product: A' = (Ax * Bx, Ay * By, Az * Bz)
  auto inline operator*(const Vector &b) const {
    return Vector(x * b.x, y * b.y, z * b.z);
  }

  // Dot product: x = A . B
  auto inline operator^(const Vector &b) const {
    // Dot product uses the forth component.
    return x * b.x + y * b.y + z * b.z + w * b.w;
  }

  // Cross product: A' = A x B
  auto inline operator|(const Vector &b) const {
    return Vector(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.z);
  }

  // Equality: A == B
  auto inline operator==(const Vector &b) const {
    return x == b.x && y == b.y && z == b.z;
  }

  // Inequality: A != B
  auto inline operator!=(const Vector &b) const { return !(*this == b); }

  // Length of vector: |A| = sqrt(x^2 + y^2 + z^2)
  auto inline size() const { return sqrt(x * x + y * y + z * z); }

  // Product of components: x * y * z
  auto inline product() const { return x * y * z; }

  // Sum of components: x + y + z
  auto inline sum() const { return x + y + z; }

  // Normalise: A' = A / |A|
  auto inline normalise() const { return *this / size(); }
};

// A 4x4 matrix. Matrices are immutable, and while declared row-wise,
// they store both row-wise and column-wise vectors internally for
// quick row or column-wise indexing.
class Matrix {
 public:
  // Row-wise vectors.
  const Vector r[4];
  // Column-wise vectors.
  const Vector c[4];

  inline Matrix(const Vector r1, const Vector r2, const Vector r3,
                const Vector r4)
      : r{r1, r2, r3, r4},
        c{Vector(r1.x, r2.x, r3.x, r4.x), Vector(r1.y, r2.y, r3.y, r4.y),
          Vector(r1.z, r2.z, r3.z, r4.z), Vector(r1.w, r2.w, r3.w, r4.w)} {}

  // Matrix multiplication.
  Matrix inline operator*(const Matrix &b) const {
    return Matrix(
        Vector(r[0] ^ b.c[0], r[0] ^ b.c[1], r[0] ^ b.c[2], r[0] ^ b.c[3]),
        Vector(r[1] ^ b.c[0], r[1] ^ b.c[1], r[1] ^ b.c[2], r[1] ^ b.c[3]),
        Vector(r[2] ^ b.c[0], r[2] ^ b.c[1], r[2] ^ b.c[2], r[2] ^ b.c[3]),
        Vector(r[3] ^ b.c[0], r[3] ^ b.c[1], r[3] ^ b.c[2], r[3] ^ b.c[3]));
  }

  // Matrix by vector multiplication.
  Vector inline operator*(const Vector &b) const {
    // Pad the "w" component.
    const Vector v = Vector(b.x, b.y, b.z, 1);
    return Vector(r[0] ^ v, r[1] ^ v, r[2] ^ v, r[3] ^ v);
  }

  // Scalar multiplication.
  Matrix inline operator*(const Scalar a) const {
    return Matrix(r[0] * a, r[1] * a, r[2] * a, r[3] * a);
  }
};

// A translation matrix.
class Translation : public Matrix {
 public:
  inline Translation(const Scalar x, const Scalar y, const Scalar z)
      : Matrix(Vector(1, 0, 0, x), Vector(0, 1, 0, y), Vector(0, 0, 1, z),
               Vector(0, 0, 0, 1)) {}
  explicit inline Translation(const Vector &t)
      : Matrix(Vector(1, 0, 0, t.x), Vector(0, 1, 0, t.y), Vector(0, 0, 1, t.z),
               Vector(0, 0, 0, 1)) {}
};

// A scale matrix.
class Scale : public Matrix {
 public:
  inline Scale(const Scalar x, const Scalar y, const Scalar z)
      : Matrix(Vector(x, 0, 0, 0), Vector(0, y, 0, 0), Vector(0, 0, z, 0),
               Vector(0, 0, 0, 1)) {}

  explicit inline Scale(const Vector &w)
      : Matrix(Vector(w.x, 0, 0, 0), Vector(0, w.y, 0, 0), Vector(0, 0, w.z, 0),
               Vector(0, 0, 0, 1)) {}
};

// A rotation matrix about the X axis.
class RotationX : public Matrix {
 public:
  explicit inline RotationX(const Scalar theta)
      : Matrix(Vector(1, 0, 0, 0),
               Vector(0, deg::cos(theta), -deg::sin(theta), 0),
               Vector(0, deg::sin(theta), deg::cos(theta), 0),
               Vector(0, 0, 0, 1)) {}
};

// A rotation matrix about the Y axis.
class RotationY : public Matrix {
 public:
  explicit inline RotationY(const Scalar theta)
      : Matrix(Vector(deg::cos(theta), 0, deg::sin(theta), 0),
               Vector(0, 1, 0, 0),
               Vector(-deg::sin(theta), 0, deg::cos(theta), 0),
               Vector(0, 0, 0, 1)) {}
};

// A rotation matrix about the Z axis.
class RotationZ : public Matrix {
 public:
  explicit inline RotationZ(const Scalar theta)
      : Matrix(Vector(deg::cos(theta), -deg::sin(theta), 0, 0),
               Vector(deg::sin(theta), deg::cos(theta), 0, 0),
               Vector(0, 0, 1, 0), Vector(0, 0, 0, 1)) {}
};

// Yaw, pitch, roll rotation.
Matrix inline rotation(const Scalar x, const Scalar y, const Scalar z) {
  return RotationZ(z) * RotationY(y) * RotationX(x);
}

}  // namespace rt

#endif  // RT_MATH_H_
