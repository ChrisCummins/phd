#!/usr/bin/env bash
#
# A smoke test to ensure that program runs without crashing. This doesn't test
# the results of execution, other than the return code.

ME_DB="datasets/me_db/me_db"
INBOX_PATH="datasets/me_db/tests/test_inbox"
TMPDIR="$(mktemp -d)"
DB_PATH="$TMPDIR/me_db_smoke_test.db"

# Tidy up.
cleanup() {
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

set -eux

# Run it once.
"$ME_DB" --inbox "$INBOX_PATH" --v=1 --db="sqlite:///$DB_PATH"
# Run it again, replacing the existing database.
"$ME_DB" --inbox "$INBOX_PATH" --v=1 --db="sqlite:///$DB_PATH" --replace_existing
