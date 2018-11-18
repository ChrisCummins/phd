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
     --push movies --nodelete --nodry_run $@)
  (cd ~/phd && bazel run //system/machines:machine -- \
     --machine=$HOME/phd/system/machines/ryangosling/ryangosling.pbtxt \
     --push tv --nodelete --nodry_run $@)
}
