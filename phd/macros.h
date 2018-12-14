// This file defines utility macros for working with C++.
#pragma once

#include "phd/private/macros_impl.h"

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef DEBUG
#error "C preprocessor macro DEBUG() already defined!"
#endif

#ifdef INFO
#error "C preprocessor macro INFO() already defined!"
#endif

#ifdef WARN
#error "C preprocessor macro WARN() already defined!"
#endif

#ifdef ERROR
#error "C preprocessor macro ERROR() already defined!"
#endif

#ifdef FATAL
#error "C preprocessor macro FATAL() already defined!"
#endif

// Log the given message with varying levels of severity. The arguments should
// be a format string. A newline is appended to the message when printed.
#define DEBUG(...) LOG_WITH_PREFIX("D", __VA_ARGS__)
#define INFO(...) LOG_WITH_PREFIX("I", __VA_ARGS__)
#define WARN(...) LOG_WITH_PREFIX("W", __VA_ARGS__)
#define ERROR(...) LOG_WITH_PREFIX("E", __VA_ARGS__)
// Terminate the program with exit code 1, printing the given message to
// stderr, followed by a stack trace. The arguments should be a format string.
// A newline is appended to the message when printed. Code to dump a stack
// trace by @tgamblin at: https://stackoverflow.com/a/77336
#define FATAL(...) \
  { \
    LOG_WITH_PREFIX("F", __VA_ARGS__); \
    void *array[10]; \
    size_t size; \
    size = backtrace(array, 10); \
    backtrace_symbols_fd(array, size, STDERR_FILENO); \
    exit(1); \
  }

#ifdef CHECK
#error "C preprocessor macro CHECK() already defined!"
#endif

// Check that `conditional` is true else fail fatally.
#define CHECK(conditional) \
  {  \
    if (!(conditional)) { \
      FATAL("CHECK(" #conditional ") failed!"); \
    } \
  }

namespace test {
namespace debug {

// Debug type:
template<typename T> struct debug_t {};

}  // debug namespace
}  // test namespace

// Macros for debugging types:
//
// Fatally crash the compiler by attempting to construct an object of
// 'type' using an unknown argument.
#define PRINT_TYPENAME(type) type _____{test::debug::debug_t<type>};
//
// Fatally crash the compiler by attempting to cast 'variable' to an
// unknown type.
#define PRINT_TYPE(variable) static_cast<test::debug::debug_t<decltype(variable)>>(variable);
