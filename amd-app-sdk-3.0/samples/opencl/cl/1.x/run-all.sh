#!/usr/bin/env bash
set -u

ROOT=$(pwd)

cmake .
make

rm -rfv /tmp/amd-logs/ logs
mkdir -p /tmp/amd-logs/good
mkdir -p /tmp/amd-logs/bad

for f in $(find . -type f -executable | grep -v CMake | grep -v run-*.sh | sort); do
     echo -n "$(tput bold)$f$(tput sgr0) ... "
     cd $(dirname $f)
     ./$(basename $f) &> /tmp/amd-logs/current.log
     ret=$?
     cd $ROOT

     if [[ $ret -ne 0 ]]; then
         echo "fail"
         mv /tmp/amd-logs/current.log /tmp/amd-logs/bad/$(basename $f).log
     else
         echo "ok"
         mv /tmp/amd-logs/current.log /tmp/amd-logs/good/$(basename $f).log
     fi
done

mv /tmp/amd-logs $ROOT/logs
