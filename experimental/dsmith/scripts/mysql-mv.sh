#!/usr/bin/env bash
#
# Rename a database
set -eux

main() {
    local old_db=$1
    local new_db=$2

    mysql -sNe "CREATE DATABASE $new_db"
    mysql $old_db -sNe 'show tables' | while read table; do
        mysql -sNe "RENAME TABLE $old_db.$table to $new_db.$table"
    done
}
main $@
