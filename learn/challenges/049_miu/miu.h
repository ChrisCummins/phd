// A solver for Douglas Hofstadter's MU puzzle.
#pragma once

#include <vector>
#include "labm8/cpp/string.h"

namespace miu {

// Solv an MU puzzle using brute force.
//
// Starting with the input string, use a breadth-first enumeration of all
// variants, stopping when the end string is enumerated, or if maxStep attempts
// have been made.
//
// Is a solution is found, all of the intermediate strings are returned.
//
// This implementation scales exponentially in both time and space with the
// number of steps.
std::vector<string> Solve(const string& start, const string& end,
                          int64_t maxStep = 0);

}  // namespace miu
