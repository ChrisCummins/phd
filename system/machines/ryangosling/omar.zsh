#!/usr/bin/env zsh

ryan_gosling_have_my_files() {
  (~/.local/bin/machines \
     --machine=$HOME/.local/var/machines/ryangosling.pbtxt \
     --push bodie_linux --delete --nodry_run $@)
}
