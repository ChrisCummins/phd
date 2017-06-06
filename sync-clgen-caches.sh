#!/usr/bin/env bash
set -eux

main() {
        local rsync="rsync -avh --exclude dirhashcache.db"

        ssh cc1 "$rsync ~/.cache/clgen/ cc2:~/.cache/clgen/"
        ssh cc1 "$rsync ~/.cache/clgen/ cc3:~/.cache/clgen/"

        ssh cc2 "$rsync ~/.cache/clgen/ cc1:~/.cache/clgen/"
        ssh cc2 "$rsync ~/.cache/clgen/ cc3:~/.cache/clgen/"

        ssh cc3 "$rsync ~/.cache/clgen/ cc1:~/.cache/clgen/"
        ssh cc3 "$rsync ~/.cache/clgen/ cc2:~/.cache/clgen/"
}
main $@
