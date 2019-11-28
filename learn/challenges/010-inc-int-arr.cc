/*
 * Given a number, expressed as an array of digits, provide a function
 * to increment it.
 */
#include <iostream>
#include <string>

#include <phd/test>

char* inc_int_str(char* str) {
  size_t len{0};
  char* cc;

  // Find last non-empty character.
  for (cc = str; *cc >= '0' && *cc <= '9'; ++cc, ++len)
    ;  // NOLINT(whitespace/semicolon)
  cc = &str[len - 1];

  while (true) {
    ++*cc;

    // If the value has overflowed, reset and move to next char.
    if (*cc == '9' + 1) {
      *cc = '0';

      // If we're at the last character, copy the digits values along
      // one, otherwise just move to the next character to increment.
      if (cc == str) {
        for (cc = str + len; cc != str; --cc) {
          *cc = *(cc - 1);
        }
        *cc = '1';
        break;
      } else {
        --cc;
      }
    } else {
      break;
    }
  }

  return str;
}

TEST(incr_int_arr, str) {
  char str1[] = "0", str2[] = "01", str3[] = "002", str4[] = "9 ",
       str5[] = "1239", str6[] = "999 ";

  ASSERT_STREQ("1", inc_int_str(str1));
  ASSERT_STREQ("02", inc_int_str(str2));
  ASSERT_STREQ("003", inc_int_str(str3));
  ASSERT_STREQ("10", inc_int_str(str4));
  ASSERT_STREQ("1240", inc_int_str(str5));
  ASSERT_STREQ("1000", inc_int_str(str6));
}

PHD_MAIN()
