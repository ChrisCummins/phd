#!/usr/bin/env bash
set -eux

logfile=~/src/project_b/data/bug-reports/.logs/$(date -Idate).txt
find -L ~/src/project_b/data/bug-reports -type f > $logfile

wc -l $logfile
