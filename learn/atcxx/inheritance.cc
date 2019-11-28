#include <iostream>

// Overriding virtual:

class Base {
 public:
  virtual ~Base() {}
  virtual void foo() const { std::cout << '0'; }
  virtual void bar() const {
    throw std::runtime_error("nothing should call this");
  }
};

class D1 : public Base {
 public:
  // no "virtual" specifier (though it is still virtual). "override" specifier
  void foo() const override { std::cout << '1'; }
};

class D2 : public Base {
 public:
  // no "override" specifier (though it is still overriding).
  virtual void foo() const { std::cout << '2'; }

  virtual void bar() const { std::cout << "D2"; }
};

// "struct" rather than class
struct D3 : public D2 {
  virtual void foo() const { std::cout << '3'; }

  virtual void bar() const { std::cout << "D3"; }
};

// Struct inheritance: (because why not)

struct mystruct {
  int data1 = 1;
  int data2 = 2;
};

struct struct2 : mystruct {
  int data1 = 10;
};

class notastruct : public struct2 {
 public:
  int data2 = 100;
};

void call_foo(const Base& b) { b.foo(); }

void call_bar(const Base& b) { b.bar(); }

int main() {
  Base b;
  D1 d1;
  D2 d2;
  D3 d3;

  b.foo();
  d1.foo();
  d2.foo();
  d3.foo();
  std::cout << std::endl << std::endl;

  call_foo(b);
  call_foo(d1);
  call_foo(d2);
  call_foo(d3);
  std::cout << std::endl << std::endl;

  d2.bar();
  std::cout << ' ';
  d3.bar();
  std::cout << std::endl << std::endl;

  call_bar(d2);
  call_bar(d3);
  std::cout << std::endl << std::endl;

  struct mystruct s1;
  struct struct2 s2;
  notastruct s3;

  std::cout << s1.data1 << ' ' << s1.data2 << std::endl
            << s2.data1 << ' ' << s2.data2 << std::endl
            << s3.data1 << ' ' << s3.data2 << std::endl;

  return 0;
}
