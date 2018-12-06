// A program that fails with an error if not called with an argument.
#include "phd/macros.h"

int main(int argc, char** argv) {
  CHECK(argc == 2);
}
