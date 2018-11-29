// This file defines utility macros for working with C++.
#pragma once

#include "absl/strings/str_format.h"

#include <iostream>

#ifdef FATAL
#error "C preprocessor macro FATAL() already defined!"
#endif

// Terminate the program with exit code 1, printing the given message to
// stderr. The arguments should be a format string. A newline is appended
// to the message when printed.
#define FATAL(...) \
  { \
    std::cerr << "fatal: [" __FILE__ ":" << __LINE__ << "] " \
              << absl::StrFormat(__VA_ARGS__) << std::endl; \
    exit(1); \
  }

#ifdef CHECK
#error "C preprocessor macro CHECK() already defined!"
#endif

// Check that `conditional` is true else fail fatally.
#define CHECK(conditional) \
  {  \
    if (!conditional) { \
      FATAL("CHECK(" #conditional ") failed!"); \
    } \
  }
