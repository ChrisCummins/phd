/*
 * A palindrome is a word, phrase, number, or other sequence of
 * characters which reads the same backward or forward. Allowances may
 * be made for adjustments to capital letters, punctuation, and word
 * dividers. Examples in English include “A man, a plan, a canal,
 * Panama!”, “Amor, Roma”, “race car”, “stack cats”, “step on no
 * pets”, “taco cat”, “put it up”, “Was it a car or a cat I saw?” and
 * “No ‘x’ in Nixon”.
 *
 * Source: https://rcrowley.org/2010/01/06/things-unix-can-do-atomically.html
 */
#include <assert.h>

#define MAX_INPUT_LEN 1024
static char buf[MAX_INPUT_LEN];

// Return 0 if 'str' is a palindrome, else
int is_palindrome(const char* const str) {
  const char* c = str;
  char* o = buf;

  // Copy str into buf, converting to lowercase and discard word non
  // a-z characters.
  while (*c) {
    if (*c >= 'a' && *c <= 'z')
      *o++ = *c;
    else if (*c >= 'A' && *c <= 'Z')
      *o++ = *c - 'A' + 'a';

    *o = '\0';
    c++;
  }

  // Reset pointers
  const char* left = buf;
  const char* right = o - 1;

  // Iterate over [0:n/2) characters
  while (left < right) {
    if (*left == *right) {
      left++;
      right--;
    } else {
      return 1;
    }
  }

  return 0;
}

#define TEST(x) assert(!(x))

int main(int argc, char** argv) {
  // Check for empty string
  if (argc != 2) return -1;

  // Sanity checks
  TEST(!is_palindrome("abcd"));
  TEST(!is_palindrome("abcde"));
  TEST(!is_palindrome(" aB! acde"));

  TEST(is_palindrome("A man, a plan, a canal, Panama!"));
  TEST(is_palindrome("No 'x' in Nixon"));
  TEST(is_palindrome("Amor, Roma"));
  TEST(is_palindrome(""));

  return is_palindrome(argv[1]);
}
