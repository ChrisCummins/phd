#include "learn/daily/d181211_matrix_multiplication.h"

#include "phd/logging.h"

#include <boost/thread/thread.hpp>

namespace phd {
namespace learn {


Vector ColumnVector(const Matrix& matrix, const int column) {
  CHECK(column >=0 && column < matrix.size2());

  Vector result(matrix.size1());
  for (int i = 0; i < result.size(); ++i) {
    result[i] = matrix(i, column);
  }

  return result;
}

Vector RowVector(const Matrix& matrix, const int row) {
  CHECK(row >= 0 && row < matrix.size1());

  Vector result(matrix.size2());
  for (int i = 0; i < result.size(); ++i) {
    result[i] = matrix(row, i);
  }

  return result;
}

Scalar DotProduct(const Vector& a, const Vector& b) {
  CHECK(a.size() == b.size());

  Scalar ret = 0;
  for (int i = 0; i < a.size(); ++i) {
    ret += a[i] * b[i];
  }

  return ret;
}

// A callable class that computes a single element in a matrix multiplication.
// This is the parallelized work unit.
class MatrixMultiplicationWorkUnit {
 public:
  MatrixMultiplicationWorkUnit(const Matrix& a, const Matrix& b,
                               const int row, const int col, Matrix* output)
      : a_(a), b_(b), row_(row), col_(col), output_(output) {};

  void operator()() {
    (*output_)(row_, col_) = DotProduct(
        RowVector(a_, row_), ColumnVector(b_, col_));
  }

 private:
  const Matrix& a_;
  const Matrix& b_;
  const int row_;
  const int col_;
  Matrix*const output_;
};

Matrix Multiply(const Matrix& a, const Matrix&b) {
  CHECK(a.size2() == b.size1());

  Matrix result(a.size1(), b.size2());

  // Each element in the matrix will be computed in a separate thread. Keep
  // a list of the threads created to join() on.
  std::vector<boost::thread> threads;
  threads.reserve(a.size1() * b.size2());

  // Create a thread to process each element in the matrix.
  for (int row = 0; row < result.size1(); ++row) {
    for (int col = 0; col < result.size2(); ++col) {
      threads.push_back(boost::thread(
          MatrixMultiplicationWorkUnit(a, b, row, col, &result)));
    }
  }

  // Block until all threads have completed.
  for (auto& thread : threads) {
    thread.join();
  }

  return result;
}

}  // namespace learn
}  // phd
