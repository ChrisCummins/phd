#!/usr/bin/env bash
#
# Self-contained batch testing for OpenCL compilers.
#
# What it does
# ------------
# 1. Creates an isolated virtual environment to install python deps in.
# 2. Runs batch testing of CLSmith, CLgen, and GitHub OpenCL kernels for a
#    predetermined length of time.
# 3. Stores results in a MySQL database running on a local server.
# 4. TODO: dumps MySQL tables somewhere that Chris can download them.
#
# Usage
# -----
# Set TIMEOUT variable to the desired number of seconds to run for, e.g.
# to test for one, hour:
#
#     $ export TIMEOUT=3600
#     $ ./jenkins.sh
#
# Requirements
# ------------
# 1. Python 3.6
#      Install on ubuntu 14.04/16.04 from ppa:jonathonf/python-3.6
# 2. MySQL server and cli
#      Provided in ubuntu packages 'mysql-server mysql-client'
# 3. MySQL user
#      MySQL server must be running, and configured with a user with all
#      permissions. For example, to configure for a user 'cec':
#
#      From MySQL prompt, run:
#        > CREATE USER 'cec'@'localhost' IDENTIFIED BY 'password1234';
#        > GRANT ALL PRIVILEGES ON *.* TO 'cec'@'localhost' WITH GRANT OPTION;
#
#      Then create a ~/.my.cnf file:
#        $ cat ~/.my.cnf
#        [mysql]
#        user=cec
#        password=password1234
#
#        [mysqladmin]
#        user=cec
#        password=password1234
#
#        [mysqldump]
#        user=cec
#        password=password1234
# 4. Build requirements for CLSmith <https://github.com/ChrisCummins/CLSmith>
#    cmake, make, etc ...
#
set -eux

# default runtime is 1 minute:
TIMEOUT=${TIMEOUT:=60}

clone_git_repo() {
    local url="$1"
    local destination="$2"
    local version="$3"

    if [[ ! -d "$destination" ]]; then
        echo "cloning $url -> $destination"
        git clone --recursive "$url" "$destination"
    fi

    if [[ ! -d "$destination/.git" ]]; then
        echo "failed: cloning repo $url to $destination" >&2
        echo "error:  $destination/.git does not exist" >&2
        exit 1
    fi

    cd "$destination"
    local target_hash="$(git rev-parse $version 2>/dev/null)"
    local current_hash="$(git rev-parse HEAD)"
    if [[ "$current_hash" != "$target_hash" ]]; then
        echo "setting repo version $destination to $version"
        git fetch --all
        git reset --hard "$version"
    fi
    cd - &>/dev/null
}


main() {
    mysqladmin status  # check MySQL server is live

    # build CLSmith
    make -C lib

    # local python environment
    test -d env || virtualenv -p python3.6 env
    source env/bin/activate
    pip install --only-binary=numpy 'numpy>=1.12.1'
    clone_git_repo git@github.com:ChrisCummins/cldrive.git env/cldrive \
        9f3aeef4b5e3418a1bc4e976b8fad51b0a30e5c4
    (cd env/cldrive && pip install -r requirements.txt && python setup.py install)
    pip install -r requirements.txt

    # setup MySQL db
    mysql -e 'CREATE DATABASE IF NOT EXISTS project_b' project_b
    mkdir -pv tmp/tables
    cp difftest/programs.tar.bz2 tmp/tables
    cd tmp/tables
    tar xjvf programs.tar.bz2
    rm programs.tar.bz2
    for tabledmp in $(ls); do
        echo "restoring $tabledmp"
        mysql project_b < $tabledmp
    done
    cd ../..
    rm -rv tmp/tables

    # run batch testing
    # TODO: If there are multiple OpenCL devices, we'll need to either be able
    # to select a specific device (perhaps by name), or iterate over all of
    # them.
    cldrive --clinfo
    platform_id=0
    device_id=0

    cd difftest
    set +e
    timeout $TIMEOUT ./run-programs.py --hostname localhost $platform_id $device_id
    ret=$?
    set -e
    cd ..

    # timeout signal is 124
    if [[ $ret -ne 124 ]] && [[ $ret -ne 0 ]]; then
        echo "$0: failed with exitstatus $ret" >&2
        exit $ret
    fi

    # export MySQL results
    mysqldump project_b > results.mysql
    tar cjvf results.tar.bz2 results.mysql
    rm -v results.mysql
    # TODO: scp results.tar.bz2 XXX
}
main $@
