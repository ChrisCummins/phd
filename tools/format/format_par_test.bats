source labm8/sh/test.sh

BIN="$(DataPath phd/tools/format/format.par)"

setup() {
  mkdir -p "$TEST_TMPDIR/src/java" "$TEST_TMPDIR/src/py"
  touch "$TEST_TMPDIR/src/py/hello.py"
  touch "$TEST_TMPDIR/src/java/Hello.java"
}

@test "run help" {
  # --help returns non-zero returncode, so just grep for a line of expected
  # output.
  "$BIN" --help | grep "format <path ...>"
  "$BIN" --helpfull | grep "format <path ...>"
}

@test "run version" {
  "$BIN" --version
}

@test "format Python" {
  "$BIN" "$TEST_TMPDIR/src/py/hello.py"
}

@test "format Java" {
  "$BIN" "$TEST_TMPDIR/src/java/Hello.java"
}
