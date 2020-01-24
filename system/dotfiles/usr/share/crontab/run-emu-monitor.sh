#!/usr/bin/env bash
#
# Start netdata daemon.
#
# ****************************************************************************
# *                               Configuration                              *
# ****************************************************************************
export LMK="/usr/local/bin/lmk -e"
export LMK_TO="chrisc.101@gmail.com"

export EMU="/usr/local/bin/emu"

export EMU_PORT=65337

# ****************************************************************************
# *                                  Program                                 *
# ****************************************************************************
set -x

cd /
sudo $EMU monitor --port $EMU_PORT --host 0.0.0.0
ret=$?
echo "emu monitor exited with returncode $ret" | $LMK -
