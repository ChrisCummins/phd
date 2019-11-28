#include <iostream>

#define STR(x) #x
#define PRINT(x) std::cout << STR(x) << " = " << x << std::endl

int rvalue_ret() { return 17; }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
int&& rrvalue_ret() {
  return 17;  // NOLINT
}
#pragma GCC diagnostic pop

int main() {
  int x;
  int l = rvalue_ret();
  int&& rr = rrvalue_ret();

  x = rr;

  PRINT(l);
  PRINT(rr);
  PRINT(rrvalue_ret());
  PRINT(x);

  return 0;
}
