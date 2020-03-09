#include "learn/challenges/049_miu/miu.h"

#include "absl/container/flat_hash_set.h"

#include <deque>
#include <iostream>
#include <vector>

namespace miu {

std::vector<string> Solve(const string& start, const string& end,
                          int64_t maxStep) {
  std::deque<std::vector<string>> q;
  absl::flat_hash_set<string> visited;

  q.push_back({start});

  int64_t step = 0;

  // Where the completed path is stored, once found.
  std::vector<string> completePath;

  while (!q.empty()) {
    ++step;

    // Terminate after a fixed number of steps.
    if (maxStep && step > maxStep) {
      break;
    }

    const std::vector<string>& path = q.front();
    const string& current = path[path.size() - 1];
    visited.insert(current);

    // Print a progress update.
    std::cout << "\rstep=" << step << ", q size=" << q.size()
              << ", path length=" << path.size();

    // Check if done.
    bool done = current.size() == end.size();
    if (done) {
      for (size_t i = 0; i < current.size(); ++i) {
        if (current[i] != end[i]) {
          done = false;
          break;
        }
      }
    }
    if (done) {
      completePath = path;
      break;
    }

    // Rule 1. If you possess a string whose last letter is I, you can add a
    // U at the end.
    if (current[current.size() - 1] == 'I') {
      const string rule1 = current + "U";
      if (visited.find(rule1) == visited.end()) {
        q.push_back(path);
        q.back().push_back(rule1);
      }
    }

    // Rule 2. Suppose you have Mx. Then you may add Mxx to your collection.
    const string rule2 = current + current.substr(1);
    if (visited.find(rule2) == visited.end()) {
      q.push_back(path);
      q.back().push_back(rule2);
    }

    // Rule 3. If III occurs in one of the strings in your collection, you may
    // make a new string with U in place of III.
    for (size_t i = 3; i < current.size(); ++i) {
      if (current[i] == 'I' && current[i - 1] == 'I' && current[i - 2] == 'I') {
        const string rule3 =
            current.substr(0, i - 2) + "U" + current.substr(i + 1);
        if (visited.find(rule3) == visited.end()) {
          q.push_back(path);
          q.back().push_back(rule3);
        }
      }
    }

    // Rule 4. If UU occurs inside one of your strings, you can drop it.
    for (size_t i = 2; i < current.size(); ++i) {
      if (current[i] == 'U' && current[i - 1] == 'U') {
        const string rule4 = current.substr(0, i - 1) + current.substr(i + 1);
        if (visited.find(rule4) == visited.end()) {
          q.push_back(path);
          q.back().push_back(rule4);
        }
      }
    }

    q.pop_front();
  }
  std::cout << "\n";

  if (completePath.size()) {
    std::cout << "Solution found:\n";
    for (size_t i = 1; i < completePath.size(); ++i) {
      std::cout << "  " << i << " " << completePath[i - 1] << " -> "
                << completePath[i] << "\n";
    }
  } else {
    std::cout << "No solution found!\n";
  }

  return completePath;
}

}  // namespace miu
