#!/usr/bin/env bash
#
# dotfiles installation script
#
# Copyright 2016-2020 Chris Cummins <chrisc.101@gmail.com>.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
usage() {
  echo "usage: $0 [-v|--verbose] [outdated]"
}

version() {
  echo "dotfiles $(git rev-parse --short HEAD)"
}

# path to this repo
dotfiles="$HOME/.dotfiles"
# path to private files
private="$HOME/Dropbox/Shared"

echo_ok() {
  local msg="$@"
  echo -e "$(tput bold)$@$(tput sgr0)"
}

echo_error() {
  local msg="$@"
  echo -e "$(tput bold)$(tput setaf 1)$@$(tput sgr0)" >&2
}

update_brew() {
  if [[ "$(uname)" == "Darwin" ]]; then
    brew update &>/dev/null

    if [[ -n "$OUTDATED" ]]; then
      echo_ok "\nbrew:"
      brew outdated | sed 's/^/  /'
      echo_ok "\nbrew casks:"
      brew cask outdated | sed 's/^/  /'
    else
      brew upgrade
      brew cask upgrade
      brew cleanup
      brew cask cleanup
    fi
  fi
}

parse_args() {
  set -e
  if [[ "$1" == "-v" ]] || [[ "$1" == "--verbose" ]]; then
    set -x
    shift
  elif [[ "$1" == "--version" ]]; then
    version
    exit 0
  elif [[ "$1" == "outdated" ]]; then
    export OUTDATED="1"
  elif [[ -n "$1" ]]; then
    usage >&2
    exit 1
  fi

  if [[ "$OUTDATED" != "1" ]]; then
    export OUTDATED=""
  fi

  set -u
}

main() {
  parse_args $@
  echo_ok "dotfiles $(git rev-parse --short HEAD)"

  if ! [[ "$(uname)" == "Darwin" ]]; then
    echo_error "Only macOS is supported"
    exit 1
  fi

  update_brew
}
main $@
