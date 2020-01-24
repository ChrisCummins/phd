#!/usr/bin/env bash
#
# Backup server.
#
# ****************************************************************************
# *                               Configuration                              *
# ****************************************************************************
export USER="cec"
export JOB_TIMEOUT=21600 # 6 hrs
export LMK="/usr/local/bin/lmk"
export LMK_TO="chrisc.101@gmail.com"
export EMU_LOGDIR="/var/log/emu"
export OLDEST_LOG_DAYS=31 # keep logs for this many days

export CRONTAB_BACKUP=~/.local/etc/crontab.txt # crontab is exported to here
export RM_DSSTORE=~/.local/bin/rm-dsstore
export EMU="/usr/local/bin/emu"

# ****************************************************************************
# *                                  Program                                 *
# ****************************************************************************
set -eux

if [[ -z "${1:-}" ]]; then
  # create log directory, and delete old logs
  sudo mkdir -pv "$EMU_LOGDIR"
  sudo chown $USER -R "$EMU_LOGDIR"
  sudo find "$EMU_LOGDIR" -mtime +$OLDEST_LOG_DAYS -exec rm -v {} \;

  # entry point
  $LMK --only-errors "timeout $JOB_TIMEOUT $0 --porcelain" 2>&1 | tee "$EMU_LOGDIR/$(date '+%Y-%m-%d-%H%M%S').txt"
else
  # main script
  df -h

  # backup crontab
  mkdir -pv ~/.local/etc
  crontab -l >$CRONTAB_BACKUP

  # sudo $RM_DSSTORE "/home/$USER"

  # create new backup
  cd /
  sudo -E $EMU push --verbose --ignore-errors
  echo "done"
fi
