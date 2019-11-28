/*
 * Mixed based arithmetic.
 */
#include <phd/test>

#include <string>

enum ctype { digit = 0, az = 1, AZ = 2 };

std::string increment(std::string c) {
  char* cc = &c[c.length() - 1];
  ctype type;

  while (true) {
    // determine type
    if (*cc >= '0' && *cc <= '9')
      type = ctype::digit;
    else if (*cc >= 'a' && *cc <= 'z')
      type = ctype::az;
    else
      type = ctype::AZ;

    ++*cc;

    // If the value has overflowed, reset and move to next char.
    if (*cc == ':' || *cc == '{' || *cc == '[') {
      // set to zero.
      if (type == ctype::digit)
        *cc = '0';
      else if (type == ctype::az)
        *cc = 'a';
      else
        *cc = 'A';

      // If we're at the last character, insert a new one, otherwise
      // just move to the next character to increment.
      if (cc == &c[0]) {
        if (type == ctype::digit)
          c.insert(c.begin(), '1');
        else if (type == ctype::az)
          c.insert(c.begin(), 'a');
        else
          c.insert(c.begin(), 'A');
        break;
      } else {
        --cc;
      }
    } else {
      break;
    }
  }

  return c;
}

std::string increment(std::string c, unsigned int i) {
  while (i--) c = increment(c);
  return c;
}

TEST(Counter, increment) {
  ASSERT_EQ("m", increment("a", 12));
  ASSERT_EQ("7", increment("2", 5));
  ASSERT_EQ("aaA0", increment("zZ9", 1));
  ASSERT_EQ("AAa0", increment("Zz9", 1));
  ASSERT_EQ("BAa0", increment("Zz9", 6761));

  ASSERT_EQ("gXrmc931", increment("gXrbk539", 278392));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  const auto ret = RUN_ALL_TESTS();
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return ret;
}
