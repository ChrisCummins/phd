#!/usr/bin/env bash
#
# mkrelease.sh - push a new python package version
#
# Uses version numbering scheme <major>.<minor>.<micro>.
#
# Copyright (C) 2015, 2016 Chris Cummins.
#
# This file is part of labm8.
#
# Labm8 is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Labm8 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with labm8.  If not, see <http://www.gnu.org/licenses/>.
#
set -eu

# Print program usage
usage() {
    echo "Usage: $0 <version>"
    echo
    echo "Current version is: $(get_current_version)."
}

# Lookup the root directory for the project. If unable to locate root,
# exit script.
#
#     @return The absolute path to the project root directory
get_project_root() {
    while [[ "$(pwd)" != "/" ]]; do
        if test -f setup.py; then
            pwd
            return
        fi
        cd ..
    done

    echo "fatal: Unable to locate project base directory." >&2
    exit 3
}

# Given a version string in the form <major>.<minor>.<micro>, return
# the major component.
#
#     @return Major component as an integer, e.g. '5'
get_major() {
    echo "$1" | sed -r 's/^([0-9]+)\.[0-9]+\.[0-9]+$/\1/'
}

# Given a version string in the form <major>.<minor>.<micro>, return
# the minor component.
#
#     @return Minor component as an integer, e.g. '5'
get_minor() {
    echo "$1" | sed -r 's/^[0-9]+\.([0-9]+)\.[0-9]+$/\1/'
}

# Given a version string in the form <major>.<minor>.<micro>, return
# the micro component.
#
#     @return Micro component as an integer, e.g. '5'
get_micro() {
    echo "$1" | sed -r 's/^[0-9]+\.[0-9]+\.([0-9]+)$/\1/'
}

# Find and return the current version string in the form
# <major>.<minor>.<micro>
#
#     @return Current version string, e.g. '0.1.4'
get_current_version() {
    cd "$(get_project_root)"

    python ./setup.py --version
}

# Replace the project version with a new one.
#
#     @param $1 The new version string
set_new_version() {
    local new=$1

    local current="$(get_current_version)"

    cd "$(get_project_root)"

    echo "Updating version string... 'README.md'"
    sed "s/$current/$new/g" -i README.md
    git add README.md

    echo "Updating version string... 'setup.py'"
    sed "s/$current/$new/g" -i setup.py
    git add setup.py

    echo "Updating version string... 'docs/conf.py'"
    sed "s/$current/$new/g" -i docs/conf.py
    git add docs/conf.py

    echo "Updating version string... 'docs/index.rst'"
    sed "s/$current/$new/g" -i docs/index.rst
    git add docs/index.rst

    echo "Updating install scripts..."
    sed "s/$current/$new/g" -i install-cpu.sh
    sed "s/$current/$new/g" -i install-opencl.sh
    sed "s/$current/$new/g" -i install-cuda.sh
    git add install.sh
}

# Make the version bump.
#
#     @param $1 The new version string
make_version_bump() {
    local new_version=$1

    cd "$(get_project_root)"

    echo "Publishing documentation..."
    make docs-publish

    echo "Creating version bump commit... $new_version"
    git commit -m "Release $new_version" >/dev/null

    echo "Creating release tag... '$new_version'"
    git tag "$new_version"

    git push origin master
    git push origin "$new_version"
}

# Perform the new release.
#
#     @param $1 New version string
do_mkrelease() {
    local new_version=$1

    echo -n "Getting current version... "
    local current_version=$(get_current_version)
    echo "'$current_version'"

    set_new_version "$new_version"
    make_version_bump "$new_version"
}

# Given a version string in the form <major>.<minor>.<micro>, verify
# that it is correct.
#
#     @return 0 if version is valid, else 1
verify_version() {
    local version="$1"

    local major="$(get_major "$version")"
    local minor="$(get_minor "$version")"
    local micro="$(get_micro "$version")"

    test -n "$major" || return 1;
    test -n "$minor" || return 1;
    test -n "$micro" || return 1;

    return 0;
}

main() {
    set +u
    # Set debugging output if DEBUG=1
    test -n "$DEBUG" && {
        set -x
    }

    # Check for help argument and print usage
    for arg in $@; do
        if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
            usage
            exit 0
        fi
    done

    # Check for new version argument
    if test -z "$1"; then
        usage
        exit 1
    fi
    set -u

    if [ $(git diff --cached --exit-code HEAD^ > /dev/null \
          && (git ls-files --other --exclude-standard --directory \
              | grep -c -v '/$')) ]; then
        echo "Unstaged changes. Please commit them before continuing"
        exit 1
    fi

    # Sanity-check on supplied version string
    if ! verify_version "$1"; then
        echo "Invalid version string!" >&2
        exit 1
    fi

    do_mkrelease "$1"
}
main $@
