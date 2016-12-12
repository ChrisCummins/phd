#!/usr/bin/env bash
set -eu

main() {
    local db="$1"

    rm -rf normal encoded lr-encoded-corpus.txt
    mkdir -p encoded

    smith-train $db -d normal
    for f in $(find normal -type f); do
        echo $f
        nchar=$(wc -c $f | awk '{print $1}')

        for idx in $(shuf -i 10-$((nchar-2)) -n 10); do
            local outpath="encoded/$(basename $f)-$idx.cl"
            echo "  $outpath"

            smith-lr-encode $f $idx > $outpath
            cat $outpath >> lr-encoded-corpus.txt
            echo >> lr-encoded-corpus.txt
        done
    done
}
main $@

