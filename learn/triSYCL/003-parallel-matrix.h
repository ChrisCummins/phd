#ifndef PARALLEL_MATRIX_H
#define PARALLEL_MATRIX_H

#include "./trisycl.h"

//
// 2D floating point matrix class, offering (very naively implemented)
// SYCL-powered parallel operators. Provides read only access to
// elements using begin(), end(), and data().
//
class matrix {
 public:
  matrix(const size_t& nrows, const size_t& ncols, const float& val = float{})
      : _data(nrows * ncols, val), _nrows(nrows), _ncols(ncols) {}

  matrix(std::initializer_list<std::initializer_list<float>> il)
      : _data(il.size() * il.begin()->size()),
        _nrows(il.size()),
        _ncols(il.begin()->size()) {
    auto out = _data.begin();
    for (auto it1 = il.begin(); it1 != il.end(); it1++)
      for (auto it2 = it1->begin(); it2 != it1->end(); it2++) *out++ = *it2;
  }

  friend auto& operator<<(std::ostream& out, const matrix& m) {
    for (size_t j = 0; j < m._nrows; j++) {
      for (size_t i = 0; i < m._ncols; i++) {
        out << m._data.at(j * m._ncols + i) << ' ';
      }
      std::cout << '\n';
    }

    return out;
  }

  auto begin() { return _data.begin(); }
  auto begin() const { return _data.begin(); }
  auto end() const { return _data.end(); }
  auto data() const { return _data.data(); }

  float& at(const size_t y, const size_t x) {
    return _data.at(y * ncols() + x);
  }

  const float& at(const size_t y, const size_t x) const {
    return _data.at(y * ncols() + x);
  }

  size_t nrows() const { return _nrows; }
  size_t ncols() const { return _ncols; }

  //
  // Matrix addition.
  //
  matrix operator+(const matrix& rhs) const {
    if (rhs.ncols() != ncols() || rhs.nrows() != nrows())
      throw std::invalid_argument("unequal matrix sizes");

    auto out = matrix(nrows(), ncols());
    {
      cl::sycl::queue myQueue;
      cl::sycl::buffer<float, 2> dev_l(data(),
                                       cl::sycl::range<2>{nrows(), ncols()});
      cl::sycl::buffer<float, 2> dev_r(
          rhs.data(), cl::sycl::range<2>{rhs.nrows(), rhs.ncols()});
      cl::sycl::buffer<float, 2> dev_o(
          out.data(), cl::sycl::range<2>{out.nrows(), out.ncols()});

      myQueue.submit([&](cl::sycl::handler& cgh) {
        auto kl = dev_l.get_access<cl::sycl::access::read>(cgh);
        auto kr = dev_r.get_access<cl::sycl::access::read>(cgh);
        auto ko = dev_o.get_access<cl::sycl::access::write>(cgh);
        cgh.parallel_for(
            cl::sycl::range<2>{out.nrows(), out.ncols()},
            [=](const cl::sycl::id<2> i) { ko[i] = kl[i] + kr[i]; });
      });
    }

    return out;
  }

  //
  // Matrix subtraction.
  //
  matrix operator-(const matrix& rhs) const {
    if (rhs.ncols() != ncols() || rhs.nrows() != nrows())
      throw std::invalid_argument("unequal matrix sizes");

    auto out = matrix(nrows(), ncols());
    {
      cl::sycl::queue myQueue;
      cl::sycl::buffer<float, 2> dev_l(data(),
                                       cl::sycl::range<2>{nrows(), ncols()});
      cl::sycl::buffer<float, 2> dev_r(
          rhs.data(), cl::sycl::range<2>{rhs.nrows(), rhs.ncols()});
      cl::sycl::buffer<float, 2> dev_o(
          out.data(), cl::sycl::range<2>{out.nrows(), out.ncols()});

      myQueue.submit([&](cl::sycl::handler& cgh) {
        auto kl = dev_l.get_access<cl::sycl::access::read>(cgh);
        auto kr = dev_r.get_access<cl::sycl::access::read>(cgh);
        auto ko = dev_o.get_access<cl::sycl::access::write>(cgh);
        cgh.parallel_for(
            cl::sycl::range<2>{out.nrows(), out.ncols()},
            [=](const cl::sycl::id<2> i) { ko[i] = kl[i] - kr[i]; });
      });
    }

    return out;
  }

  friend bool operator==(const matrix& lhs, const matrix& rhs) {
    return lhs._data == rhs._data;
  }

 private:
  std::vector<float> _data;
  size_t _nrows;
  size_t _ncols;
};

#endif  // PARALLEL_MATRIX_H
