#!/usr/bin/env zsh

ryan_gosling_have_my_music() {
  (~/.local/bin/machines \
     --machine=$HOME/.local/var/machines/ryangosling.pbtxt \
     --push music --delete --nodry_run $@)
}

ryan_gosling_have_my_movies() {
  (~/.local/bin/machines \
     --machine=$HOME/.local/var/machines/ryangosling.pbtxt \
     --push movies,tv --nodelete --nodry_run --progress $@)
}

ryan_gosling_have_my_files() {
  (~/.local/bin/machines \
     --machine=$HOME/.local/var/machines/ryangosling.pbtxt \
     --push $(hostname) --delete --nodry_run $@)
}
