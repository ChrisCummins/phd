#!/usr/bin/env bash
#
# Create a database.
#
# Usage: db-create <db_name>
#

# --- begin labm8 init ---
f=phd/labm8/sh/app.sh
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null ||
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null ||
  source "$0.runfiles/$f" 2>/dev/null ||
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null ||
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null ||
  {
    echo >&2 "ERROR: cannot find $f"
    exit 1
  }
f=
# --- end app init ---

set -eu

SQL_DIR="$(dirname $(DataPath phd/programl/db/sql/schema.sql))"

main() {
  local dbname="${1:-programl}"

  cd $SQL_DIR

  set -x
  dropdb --if-exists "$dbname"
  createdb "$dbname"
  psql -d "$dbname" <schema.sql
}
main $@
