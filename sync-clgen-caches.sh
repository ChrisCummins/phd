#!/usr/bin/env bash
set -eux

main() {
        ssh cc1 "rsync -avh ~/.cache/clgen/ cc2:~/.cache/clgen/"
        ssh cc1 "rsync -avh ~/.cache/clgen/ cc3:~/.cache/clgen/"

        ssh cc2 "rsync -avh ~/.cache/clgen/ cc1:~/.cache/clgen/"
        ssh cc2 "rsync -avh ~/.cache/clgen/ cc3:~/.cache/clgen/"

        ssh cc3 "rsync -avh ~/.cache/clgen/ cc1:~/.cache/clgen/"
        ssh cc3 "rsync -avh ~/.cache/clgen/ cc2:~/.cache/clgen/"
}
main $@
