// Private implementation file for the macros header.
#pragma once

#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include <iostream>

#ifdef LOG_WITH_PREFIX
#error "C preprocessor macro LOG_WITH_PREFIX() already defined!"
#endif

// Format and print a logging message with the given prefix.
#define LOG_WITH_PREFIX(prefix, ...) \
  std::cerr << absl::FormatTime(prefix " %Y-%m-%d %H:%M:%S [" __FILE__ ":", \
                                absl::Now(), absl::TimeZone()) \
            << __LINE__ << "] " << absl::StrFormat(__VA_ARGS__) << std::endl
