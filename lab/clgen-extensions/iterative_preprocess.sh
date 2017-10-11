#!/usr/bin/env bash

set -ex

datadir=data/iterative_pp
mkdir -p $datadir

if [[ ! -f $datadir/pp1.db ]]; then
    clgen-create-db $datadir/pp1.db
    clgen-fetch $datadir/pp1.db ~/data/kernels/github
fi
clgen-preprocess $datadir/pp1.db
clgen-explore $datadir/pp1.db

set +x
for j in {2..3}; do
    i=$((j-1))
    echo "iteration $i"
    if [[ ! -f $datadir/pp$j.db ]]; then
        rm -fr $datadir/pp$j
        set -x
        clgen-dump $datadir/pp$i.db -d $datadir/pp$j
        clgen-create-db $datadir/pp$j.db
        clgen-fetch $datadir/pp$j.db $datadir/pp$j
        set +x
        rm -fr $datadir/pp$j
    fi

    clgen-preprocess $datadir/pp$j.db
    clgen-explore $datadir/pp$j.db
done
