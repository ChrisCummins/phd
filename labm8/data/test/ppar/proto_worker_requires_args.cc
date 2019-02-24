// A program that fails with an error if not called with an argument.
#include "phd/logging.h"

int main(int argc, char** argv) {
  CHECK(argc == 2);
}
