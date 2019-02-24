#!/usr/bin/env zsh

ryan_gosling_give_me_photos() {
  (cd ~/phd &&
   bazel run //system/machines:machine -- \
     --machine=$HOME/phd/system/machines/ryangosling/ryangosling.pbtxt \
     --pull photos --delete --nodry_run $@)
}

ryan_gosling_have_my_photos() {
  (cd ~/phd && bazel run //system/machines:machine -- \
     --machine=$HOME/phd/system/machines/ryangosling/ryangosling.pbtxt \
     --push photos --delete --nodry_run $@)
}

ryan_gosling_have_my_music() {
  (cd ~/phd && bazel run //system/machines:machine -- \
     --machine=$HOME/phd/system/machines/ryangosling/ryangosling.pbtxt \
     --push music --delete --nodry_run $@)
}

ryan_gosling_have_my_movies() {
  (cd ~/phd && bazel run //system/machines:machine -- \
     --machine=$HOME/phd/system/machines/ryangosling/ryangosling.pbtxt \
     --push movies,tv --nodelete --nodry_run --progress $@)
}

ryan_gosling_have_my_files() {
  (cd ~/phd && bazel run //system/machines:machine -- \
     --machine=$HOME/phd/system/machines/ryangosling/ryangosling.pbtxt \
     --push $(hostname) --delete --nodry_run $@)
}
