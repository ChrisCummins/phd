// Utility code for working with strings.
#pragma once

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

using std::string;

namespace phd {

// Trim a string from the left in-place.
void TrimLeft(string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

// Trim a string from the end in-place.
void TrimRight(string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// Trim a string from both ends in-place.
string& Trim(string &s) {
    TrimLeft(s);
    TrimRight(s);
    return s;
}

// Trim a string from the left.
string TrimLeftCopy(string s) {
    TrimLeft(s);
    return s;
}

// Trim a string from the right.
string TrimRightCopy(string s) {
    TrimRight(s);
    return s;
}

// Trim a string from both ends.
string TrimCopy(string s) {
    Trim(s);
    return s;
}

// Returns whether full_string ends with suffix.
bool EndsWith(const string& full_string, const string& suffix) {
  if (full_string.length() >= suffix.length()) {
    return full_string.compare(
        full_string.length() - suffix.length(), suffix.length(), suffix) == 0;
  } else {
    return false;
  }
}

// Convert a string to CamelCase. E.g. "hello world" -> "HelloWorld".
string ToCamelCase(const string& full_string) {
  // Split string into a vector of space separated components.
  auto split_on_whitespace = absl::StrSplit(
      full_string, ' ');
  std::vector<string> space_separated_components(
      split_on_whitespace.begin(), split_on_whitespace.end());

  // Convert starting letters to uppercase and append to string.
  string camel_case = "";
  for (auto component : space_separated_components) {
    component[0] = std::toupper(component[0]);
    absl::StrAppend(&camel_case, component);
  }

  return camel_case;
}

}  // namespace phd
