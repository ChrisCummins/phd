// An implementation of multithreaded matrix multiplication. The idea is not to
// be especially efficient or fast, but to be a clean implementation.
#pragma once

#include <boost/numeric/ublas/matrix.hpp>

namespace labm8 {
namespace learn {

// For convenience, use float scalar type, and Boost's matrix class template.
using Matrix = boost::numeric::ublas::matrix<float>;

using Vector = std::vector<float>;

using Scalar = float;

// Functions which return a Column or Row from a matrix. If the requested column
// or row is out of bounds, this crashes.
//
// NOTE: This is an inefficient implementation, requiring a copy of all values
// from the matrix container to a vector. For "real world" use this would
// instead construct an input iterator pair, providing a view of the values.
Vector ColumnVector(const Matrix& matrix, int column);

Vector RowVector(const Matrix& matrix, int row);

// Calculate the dot product of two vectors. The vectors must be the same size,
// else this crashes.
//
// NOTE: For "real world" use this would take iterators as arguments, rather
// than containers.
Scalar DotProduct(const Vector& a, const Vector& b);

// Multiply the two matrices and return the result. Given arguments of size
// [n,m] and [m,p], this returns a matrix of dimensions [m,p]. In case of
// invalid dimensions, this crashes.
Matrix Multiply(const Matrix& a, const Matrix& b);

}  // namespace learn
}  // namespace labm8
