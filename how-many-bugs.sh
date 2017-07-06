#!/usr/bin/env bash
set -eux

logfile=~/src/project_b/data/bug-reports/.logs/$(date -Idate).txt
find ~/src/project_b/data/bug-reports -type f -name '*.sh' > $logfile

wc -l $logfile
