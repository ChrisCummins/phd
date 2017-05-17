#!/usr/bin/env bash
host=cc1
db=project_b
tables=(CLSmithPrograms CLgenPrograms GitHubPrograms)
dstdir=~/src/project_b/data/sql/$host

set -eux

for table in ${tables[@]}; do
    ssh $host "mkdir -p /tmp/tables"
    ssh $host "mysqldump $db $table > /tmp/tables/$db-$table.mysql"
done

ssh $host "mkdir -p $dstdir"
ssh $host "cd /tmp/tables && tar cjvf $dstdir/$db-$(date '+%Y-%m-%d')-programs.tar.bz2 *"
ssh $host "rm -rfv /tmp/tables"
