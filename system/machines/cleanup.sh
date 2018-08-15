# TODO: This script will perform a deep clean of a system, removing directories
# which accumulate crap over time.

dirs=(
  /private/var/tmp/_bazel_$USER
  $(brew --cache)
)

sudo apt-get autoremove -y
which brew &>/dev/null && brew cleanup
which brew &>/dev/null && brew cask cleanup
apt-get clean
