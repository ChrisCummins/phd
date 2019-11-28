#include <initializer_list>
#include <iostream>

// A vector class

template <typename T>
class Container {
 public:
  virtual ~Container() {}
  virtual size_t size() const = 0;
};

template <typename T>
class MyVector : Container<T> {
 public:
  // Create uninitialised
  explicit MyVector(const size_t size) {
    if (size == 0) throw std::invalid_argument("0 size vector!");

    _size = size;
    _data = new T[size];
  }

  // Copy-construct from a list of values:
  explicit MyVector(const std::initializer_list<T> vals)
      : MyVector(vals.size()) {
    size_t i = 0;

    for (auto *v = vals.begin(); v != vals.end(); v++) {
      _data[i] = *v;
      i++;
    }
  }

  virtual ~MyVector() { delete[] _data; }

  T &operator[](const size_t i) {
    if (i >= size()) throw std::out_of_range("Vector::operator[]");

    return _data[i];
  }

  const T &operator[](const size_t i) const {
    if (i >= size()) throw std::out_of_range("Vector::operator[]");

    return _data[i];
  }

  size_t size() const { return _size; }

  friend std::ostream &operator<<(std::ostream &o, const MyVector<T> &v) {
    for (size_t i = 0; i < v._size; i++) o << v[i] << " ";
    return o;
  }

 private:
  T *_data;
  size_t _size;
};

// Support iteration:

template <typename T>
const T *begin(const MyVector<T> &v) {
  return v.size() ? &v[0] : nullptr;
}

template <typename T>
const T *end(const MyVector<T> &v) {
  return begin(v) + v.size();
}

// Summation operator:

template <typename T>
T &sum(const MyVector<T> &vec, T *acc) {
  for (auto &v : vec) *acc += v;

  return *acc;
}

int main(int argc, char **argv) {
  std::cout << "Hello, world!" << std::endl;

  MyVector<int> v{0, 1, 2, 3, 4};

  int i = 0;

  std::cout << "Vector contents: " << v << std::endl;
  std::cout << "Vector sum: " << sum(v, &i);

  return 0;
}
