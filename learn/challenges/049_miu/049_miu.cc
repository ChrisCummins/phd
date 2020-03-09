// A solver for Douglas Hofstadter's MU puzzle.
#include "labm8/cpp/app.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/string.h"
#include "learn/challenges/049_miu/miu.h"

DEFINE_string(start, "MI", "The starting string.");
DEFINE_string(end, "MU", "The ending string.");
DEFINE_int32(max_steps, 100000,
             "The maximum number of attempts to find a solution.");

static bool ValidateTheorem(const char* flagname, const string& s) {
  CHECK(s.size()) << "--" << flagname << " cannot be empty string";
  CHECK(s[0] == 'M') << "--" << flagname << " must start with 'M'";

  for (size_t i = 1; i < s.size(); ++i) {
    char c = s[i];
    CHECK(c == 'I' || c == 'U')
        << flagname << " contains invalid character '" << c << "'";
  }

  return true;
}

DEFINE_validator(start, &ValidateTheorem);
DEFINE_validator(end, &ValidateTheorem);

int main(int argc, char** argv) {
  labm8::InitApp(&argc, &argv);

  miu::Solve(FLAGS_start, FLAGS_end, FLAGS_max_steps);
}
