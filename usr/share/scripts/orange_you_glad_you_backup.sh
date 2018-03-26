#!/usr/bin/env bash
#
# Usage:
#     $ orange_you_glad_you_backup [rsync-options]
set -eu


backup() {
    local src="$1"
    local dst="$2"

    if [[ ! -d "$src" ]]; then
        echo "fatal: '$src' not found" >&2
        exit 1
    fi

    if [[ ! -d "$dst" ]]; then
        echo "fatal: '$dst' not found" >&2
        exit 1
    fi

    echo
    echo "================================================================="
    rsync -avh --delete "$src/" "$dst/" \
        --exclude "*.lrcat-journal" \
        --exclude "*.lrcat.lock" \
        --exclude "._.DS_Store" \
        --exclude ".com.apple.timemachine.supported" \
        --exclude ".DS_Store" \
        --exclude ".sync.ffs_db" \
        --exclude "/.DocumentRevisions-V100" \
        --exclude "/.fseventsd" \
        --exclude "/.Spotlight-V100" \
        --exclude "/.TemporaryItems" \
        --exclude "/.Trashes" \
        --exclude "/.VolumeIcon.icns" \
        --exclude "/.VolumeIcon.ico" \
        --exclude "/autorun.inf" \
        $@
    echo "================================================================="
    echo
}


main() {
    # TODO(cec): Remove the 'photos/2018' qualifier once ryangosling has
    # been synced to the new layout.
    backup "/Volumes/Orange/photos/2018/" "/Volumes/Satsuma/photos/2018/" $@
}
main $@
