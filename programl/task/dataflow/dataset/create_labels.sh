# Generate dataflow graphs.
#
# Usage:
#
#    $ bazel run -c opt //programl/task/dataflow/dataset:create_labels -- \
#        <dataset_path> [analysis ...]
#
# Where <dataset_path> is the absolute path of the dataset root, and [analysis ...] is an optional
# list of analyses. If not provided, all analyses are run.

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

ANALYZE="$(DataPath phd/programl/cmd/analyze)"
if [[ -f /usr/local/opt/findutils/libexec/gnubin/find ]]; then
  FIND=/usr/local/opt/findutils/libexec/gnubin/find
else
  FIND=find
fi

# Generate labels for all graphs using an analysis.
#
# This must be run from the root of the dataset directory.
#
# This assumes that the `analyze` command is in $PATH, e.g. by running:
#     $ bazel run -c opt //programl/cmd:install
# and adding /usr/local/opt/programl/bin to $PATH.
run_analysis() {
  local analysis="$1"

  echo "Generating $analysis labels ..."
  mkdir -p labels/"$analysis"
  # One big GNU parallel invocation to enumerate all program graphs and feed them through the
  # `analyze` command.
  "$FIND" graphs -type f -printf '%f\n' |
    sed 's/\.ProgramGraph\.pb$//' |
    parallel --resume --shuf --joblog labels/"$analysis".joblog.txt --bar \
      cat graphs/{}.ProgramGraph.pb '|' \
      "$ANALYZE" "$analysis" --stdin_fmt=pb --stdout_fmt=pb '>' \
      labels/"$analysis"/{}.ProgramGraphFeaturesList.pb
}

main() {
  cd "$1"
  shift

  if [ $# -eq 0 ]; then
    run_analysis reachability
    run_analysis domtree
  else
    for arg in "$@"; do
      run_analysis $arg
    done
  fi
}
main $@
