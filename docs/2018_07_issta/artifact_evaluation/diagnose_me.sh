#!/usr/bin/env bash

# diagnose_me.sh - Print diagnostic information.
#
# Usage:
#
#     ./diagnose_me.sh
#
set -eu

# Root of this repository.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"


# Indent the output of a command. Usage: some_command 2>&1 | indent
indent() {
  sed 's/^/    /'
}


main() {
  # Run from the repository root directory.
  cd "$ROOT"

  echo "ROOT=$ROOT"
  echo "HEAD=$(git rev-parse HEAD)"
  echo "Git status:"
  git status 2>&1 | indent
  echo "Submodule status:"
  git submodule status 2>&1 | indent

  # Print Linux version, if available.
  if [[ -f /etc/os-release ]]; then
    echo "/etc/os-release"
    cat /etc/os-release 2>&1 | indent
  fi
  # Print macOS version, if available.
  if which sw_vers &>/dev/null; then
    echo "sw_vers"
    sw_vers 2>&1 | indent
  fi

  if [[ -f "$ROOT/.env" ]]; then
    echo "Repository appears to have been bootstrapped."
  else
    echo "Repository does not appear to have been bootstrapped." >&2
    echo "Have you run ./install.sh ?" >&2
    exit 1
  fi

  if [[ -f "$ROOT/venv/phd/bin/activate" ]]; then
    echo "Python virtual environment found."
  else
    echo "Python virtual environment not found." >&2
    echo "Have you run ./install.sh ?" >&2
    exit 1
  fi

  # Activate the phd virtual environment.
  test -f "$ROOT/.env"
  # Disable unbound variable errors, since ./build/phd/.env checks whether
  # $VIRTUAL_ENV is set.
  set +u
  source "$ROOT/.env"
  # Re-enable unbound variable errors.
  set -u

  source "$ROOT/venv/phd/bin/activate"
  echo "PYTHON=$(which python)"
  echo "PYTHONPATH=$PYTHONPATH"
  echo "PYTHON_VERSION=$(python --version)"
  echo "Python packages:"
  python -m pip freeze 2>&1 | indent

  if [[ -f "$ROOT/deeplearning/deepsmith/proto/deepsmith_pb2.py" ]]; then
    echo "Python protocol buffer code is generated."
  else
    echo "Generated python protocol buffer code not found."
  fi

  if $(which clgen &>/dev/null); then
    echo "CLGEN=$(which clgen)"
  else
    echo "CLGEN=Not found."
  fi

  echo
  echo "Available OpenCL devices:"
  cat <<EOF > diagnose_me.py
from gpu import cldrive
for i, env in enumerate(cldrive.all_envs()):
  print('{}±: '.format(i + 1), env)
EOF
  python diagnose_me.py 2>&1 | indent
  rm diagnose_me.py
}
main $@
