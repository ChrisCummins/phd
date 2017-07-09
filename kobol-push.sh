#!/usr/bin/env bash
set -eux

main() {
        local rsync="rsync -avh"

        $rsync ~/src/project_b/data/ cec@cc1:~/src/project_b/data/
}
main $@
