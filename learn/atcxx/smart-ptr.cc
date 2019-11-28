#include <iostream>
#include <memory>

class X {
 public:
  X() : data(0) { std::cout << "-> X()\n"; }

  explicit X(const int n) : data(n) {
    std::cout << "-> explicit X(" << n << ")\n";
  }

  ~X() { std::cout << "-> ~X(" << data << ")\n"; }

  friend std::ostream& operator<<(std::ostream& out, const X& x) {
    out << "x = " << x.data;
    return out;
  }

 private:
  int data;
};

// Capture by reference, std::unique_ptr has no copy constructor.
void print_x(const std::unique_ptr<X>& x) { std::cout << *x << std::endl; }

// Capture by value.
void print_x(const std::shared_ptr<X> x) { std::cout << *x << std::endl; }

template <typename... Arg>
auto make_x(Arg&&... args) {
  return std::make_unique<X>(args...);
}

int main() {
  const auto a = std::make_unique<X>(10);
  const std::unique_ptr<X> b{new X(11)};
  const auto c = std::make_shared<X>(12);
  const auto d = make_x(13);

  print_x(a);
  print_x(b);
  print_x(c);
  print_x(d);

  return 0;
}
