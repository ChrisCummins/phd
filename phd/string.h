// Utility code for working with strings.
#pragma once

using std::string;

namespace phd {

// Trim a string from the left in-place.
void TrimLeft(string &s);

// Trim a string from the end in-place.
void TrimRight(string &s);

// Trim a string from both ends in-place.
string& Trim(string &s);

// Trim a string from the left.
string TrimLeftCopy(string s);

// Trim a string from the right.
string TrimRightCopy(string s);

// Trim a string from both ends.
string TrimCopy(string s);

// Returns whether full_string ends with suffix.
bool EndsWith(const string& full_string, const string& suffix);

// Convert a string to CamelCase. E.g. "hello world" -> "HelloWorld".
string ToCamelCase(const string& full_string);

}  // namespace phd
