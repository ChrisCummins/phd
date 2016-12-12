#!/usr/bin/env bash
#
# Fetch from Github
#
# Requires env variables: $GITHUB_TOKEN $GITHUB_USERNAME $GITHUB_PW
#
set -u

main() {
    local db_path="$1"

    local i=0

    while true; do
        echo "*** Running fetch, iteration $i..."

        fetch-gh "$db_path"

        i=$((i+1))
    done
}
main $@
