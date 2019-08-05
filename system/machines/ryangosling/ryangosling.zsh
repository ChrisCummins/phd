#!/usr/bin/env zsh

ryan_gosling_give_me_photos() {
  (
    ~/.local/bin/machines \
       --machine=$HOME/.local/var/machines/ryangosling.pbtxt \
       --pull staging --delete --nodry_run $@ && \
    ~/.local/bin/machines \
       --machine=$HOME/.local/var/machines/ryangosling.pbtxt \
       --pull photos --delete --nodry_run $@ && \
    ~/.local/bin/machines \
       --machine=$HOME/.local/var/machines/ryangosling.pbtxt \
       --pull photo_catalog --delete --nodry_run $@
  )
}

ryan_gosling_have_my_photos() {
  (
    ~/.local/bin/machines \
       --machine=$HOME/.local/var/machines/ryangosling.pbtxt \
       --push staging --delete --nodry_run $@ && \
    ~/.local/bin/machines \
       --machine=$HOME/.local/var/machines/ryangosling.pbtxt \
       --push photos --delete --nodry_run $@ && \
    ~/.local/bin/machines \
       --machine=$HOME/.local/var/machines/ryangosling.pbtxt \
       --push photo_catalog --delete --nodry_run $@
  )
}

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
