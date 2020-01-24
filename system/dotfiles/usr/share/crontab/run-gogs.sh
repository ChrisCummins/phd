#!/usr/bin/env bash
#
# Start gogs daemon.
# https://github.com/gogits/gogs
#
# ****************************************************************************
# *                               Configuration                              *
# ****************************************************************************
export LMK="/usr/local/bin/lmk -e"
export LMK_TO="chrisc.101@gmail.com"

export GOGS_DIR="/opt/gogs"
export GOGS="$GOGS_DIR/gogs"

# ****************************************************************************
# *                                  Program                                 *
# ****************************************************************************
set -x

cd "$GOGS_DIR"
"$GOGS" web
ret=$?
echo "exited with returncode $ret" | $LMK -
