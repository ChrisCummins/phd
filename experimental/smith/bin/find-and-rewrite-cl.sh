#!/usr/bin/env bash

main() {
    local indir="$1"
    local outdir="$2"

    echo "preprocessing $indir to $outdir ..."

    find "$indir" -name '*.cl' -print0 |
        while IFS= read -r -d $'\0' inpath; do
            outpath="$outdir/$(echo $inpath | sed -r 's,.*/(.*\.cl),\1,')"
            echo -e "$outpath"
            smith-preprocess -f "$inpath" > "$outpath"
        done
}
main $@
