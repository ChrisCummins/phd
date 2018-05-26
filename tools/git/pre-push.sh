#!/bin/sh

#
# Enforces commit push policy.
#
set -eu

remote="$1"
url="$2"

local_ref_is_priv() {
    local local_ref="$1"

    echo "$local_ref" | grep "/priv" &> /dev/null
}

local_ref_is_master() {
    local local_ref="$1"

    echo "$local_ref" | grep "master" &> /dev/null
}

url_is_priv() {
    local url="$1"

    echo "$url" | grep "priv" &> /dev/null
}

while read local_ref local_sha remote_ref remote_sha
do
    if (url_is_priv "$url"); then
        echo "Privacy: private - OK"
        exit 0
    elif (! url_is_priv "$url" && local_ref_is_master "$local_ref"); then
        echo "Privacy: public - OK"
        exit 0
    else
        echo "Privacy: public/private clash! Reason:" >&2
        echo >&2

        echo -n "  url '$url' is " >&2
        if (url_is_priv "$url"); then
            echo "PRIVATE" >&2
        else
            echo "PUBLIC" >&2
        fi

        echo -n "  local ref '$local_ref' is " >&2
        if (local_ref_is_master "$local_ref"); then
            echo "MASTER" >&2
        else
            echo "NOT MASTER" >&2
        fi

        echo >&2
        echo "Master must be public. All other branches must be private." >&2

        echo >&2
        echo "Aborting push." >&2
        exit 1
    fi
done
