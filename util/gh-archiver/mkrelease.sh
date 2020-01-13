#!/usr/bin/env bash

#
# mkrelease.sh - push a new python package version
#
# Uses version numbering scheme <major>.<minor>.<micro>.
#
# Copyright (C) 2015, 2016, 2017 Chris Cummins.
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
set -e

# List of files in which the version number should be updated
files_to_update=(
"setup.py"
)

# Branch on which releases must be made
release_branch="master"

# Version number suffix for development versions
dev_suffix=".dev0"


version_is_pep440_compliant() {
    local version="$1"

    pip install packaging &> /dev/null
    echo "$version" | python -c 'import sys; import packaging.version; packaging.version.Version(sys.stdin.read().strip())' &> /dev/null
}


git_tree_dirty() {
    [[ $(git diff --shortstat 2> /dev/null | tail -n1) != "" ]] || return 1
}


branch_is() {
    local expected_branch="$1"

    branch_name=$(git symbolic-ref -q HEAD)
    branch_name=${branch_name##refs/heads/}
    branch_name=${branch_name:-HEAD}

    [[ "$branch_name" == "$expected_branch" ]] || return 1
}


usage() {
    echo "Usage: $0 <version>"
    echo
    echo "Current version is: $1."
}


main() {
    test -n "$DEBUG" && set -x # Set debugging output if DEBUG=1
    local new_version="$1"

    set -eu

    local current_version="$(python ./setup.py --version)"

    # validate args
    for arg in $@; do
        if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
            usage "$current_version"
            exit 0
        fi
    done

    if [[ $# != 1 ]]; then
        usage "$current_version"
        exit 1
    fi

    # check that it is the root of a python package
    if [ ! -f setup.py ]; then
        echo "File setup.py not found" >&2
        exit 1
    fi

    if ! branch_is "$release_branch"; then
        echo "On the wrong branch. Must be on '$release_branch'"
        exit 1
    fi

    if git_tree_dirty; then
        echo "Unstaged changes. Please commit them before continuing"
        exit 1
    fi

    # Sanity-check on supplied version string
    if ! version_is_pep440_compliant "$new_version"; then
        echo "Version string '$1' is not compliant with PEP 440" >&2
        echo >&2
        echo "See: <https://www.python.org/dev/peps/pep-0440/>" >&2
        exit 1
    fi

    # escape current for regex
    local current_version_re="$(echo $current_version | sed -e 's/[]\/$*.^|[]/\\&/g')"

    # set new version
    for file in ${files_to_update[@]}; do
        echo "sed -r \"s/$current_version_re/$new_version/g\" -i \"$file\""
        sed -r "s/$current_version_re/$new_version/g" -i "$file"
        echo "git add \"$file\""
        git add "$file"
    done

    echo "Please review the changes staged to commit:"
    git status
    git diff --cached

    while true; do
        read -p "Commit to release $new_version? [yn] " yn
        case $yn in
            [Yy]*) break ;;
            [Nn]*) exit 1 ;;
            *) echo "Please answer yes or no." ;;
        esac
    done

    set -x
    git commit -m "Release $new_version"
    git tag "$new_version"
    git push origin "$new_version"

    # update on PyPi
    python setup.py sdist upload -r pypi

    # set dev version
    local dev_version="$new_version$dev_suffix"
    for file in ${files_to_update[@]}; do
        echo "sed -r \"s/$new_version/$dev_version/g\" -i $file"
        sed -r "s/$new_version/$dev_version/g" -i "$file"
        echo "git add \"$file\""
        git add "$file"
    done

    git commit -m "Development version bump"
    git push origin master
}

main $@
