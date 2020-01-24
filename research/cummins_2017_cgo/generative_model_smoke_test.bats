#!/usr/bin/env bats
#
# Test that `generative_models` runs without catching fire.
#
# Copyright 2017, 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

source labm8/sh/test.sh

BIN="$(DataPath phd/research/cummins_2017_cgo/generative_model)"

setup() {
  mkdir "$TEST_TMPDIR/working_dir"
  mkdir "$TEST_TMPDIR/corpus"

  # Create corpus
  cat <<EOF >"$TEST_TMPDIR/corpus/a.txt"
kernel void A(global int* a, const int b) {
  if (get_global_id(0) < b) {
    a[get_global_id(0)] *= 2;
  }
}
EOF
}

@test "generate_model smoke test" {
  "$BIN" \
    --clgen_working_dir="$TEST_TMPDIR/working_dir" \
    --clgen_corpus_dir="$TEST_TMPDIR/corpus" \
    --clgen_layer_size=8 \
    --clgen_sample_sequence_length=32 \
    --clgen_training_sequence_length=4 \
    --clgen_training_batch_size=4 \
    --clgen_max_sample_length=64 \
    --clgen_num_epochs=2 \
    --clgen_min_sample_count=5 \
    --clgen_preprocessor=deeplearning.clgen.preprocessors.opencl:Compile \
    2>&1 | tee "$TEST_TMPDIR/log.txt"
}
