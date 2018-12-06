// Utility code for working with strings.
#pragma once

#include <string>

using std::string;

namespace phd {

// Trim leading and trailing whitespace from a string.
string& Trim(string &s);

// Trim leading and trailing whitespace from a string.
string CopyAndTrim(string s);

// Trim leading whitespace from a string.
void TrimLeft(string &s);

// Trim a string from the left.
string CopyAndTrimLeft(string s);

// Trim a string from the end in-place.
void TrimRight(string &s);

// Trim a string from the right.
string CopyAndTrimRight(string s);

// Returns whether full_string ends with suffix.
bool EndsWith(const string& full_string, const string& suffix);

// Split a string on whitespace and convert the first letter of each wort
// to CamelCase. E.g. "hello world" -> "HelloWorld".
string ToCamelCase(const string& full_string);

}  // namespace phd
