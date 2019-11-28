#include <iostream>

#define PRINT(x) std::cout << #x << " = " << x << std::endl;

static unsigned int copy_constructors;
static unsigned int copy_assignments;
static unsigned int move_constructors;
static unsigned int move_assignments;

class Vector {
 public:
  explicit Vector(int s) : _elems{new double[size_t(s)]}, _size{s} {}

  ~Vector() { delete[] _elems; }

  Vector(const Vector& other) : Vector(other.size()) {
    ++copy_constructors;

    for (size_t i = 0; i < size_t(size()); ++i) _elems[i] = other[i];
  }

  Vector& operator=(const Vector& rhs) {
    ++copy_assignments;

    if (&rhs != this) {
      if (rhs.size() != size()) {
        delete[] _elems;
        _elems = new double[rhs.size()];
        _size = rhs.size();
      }

      for (size_t i = 0; i < size_t(size()); ++i) _elems[i] = rhs[i];
    }

    return *this;
  }

  Vector(Vector&& other) {
    ++move_constructors;

    _elems = other._elems;
    _size = other._size;
    other._elems = nullptr;
  }

  Vector& operator=(Vector&& rhs) {
    ++move_assignments;

    if (_elems) {
      delete[] _elems;
    }

    _elems = rhs._elems;
    _size = rhs._size;
    rhs._elems = nullptr;

    return *this;
  }

  double& operator[](int i) { return _elems[i]; }

  const double& operator[](int i) const { return _elems[i]; }

  int size() const { return _size; }

 private:
  double* _elems;
  int _size;
};

Vector operator+(const Vector& a, const Vector& b) {
  if (a.size() != b.size()) {
    std::exit(-1);
  }

  auto v = Vector(a.size());
  for (auto i = 0; i < a.size(); ++i) {
    v[i] = a[i] + b[i];
  }
  return v;
}

Vector makeVector(int size) {
  auto v = Vector(size);
  for (auto i = 0; i < size; ++i) {
    v[i] = i;
  }
  return v;
}

void printSum(Vector v) {
  auto acc = 0.0;
  for (auto i = 0; i < v.size(); ++i) {
    acc += v[i];
  }
  std::cout << "sum: " << acc << "\n";
}

int main() {
  auto v = makeVector(1024);
  auto w = makeVector(1024);
  printSum(v + w + v);

  PRINT(copy_constructors);
  PRINT(copy_assignments);
  PRINT(move_constructors);
  PRINT(move_assignments);
}
