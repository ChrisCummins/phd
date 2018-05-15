# TODO: This script will perform a deep clean of a system, removing directories
# which accumulate crap over time.

dirs=(
  /private/var/tmp/_bazel_$USER
  $(brew --cache)
)
