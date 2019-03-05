#!/usr/bin/env bash
#
# A smoke test to ensure that program runs without crashing. This doesn't test
# the results of execution, other than the return code.
set -eux

working_dir="$(mktemp -d)"

# Tidy up.
cleanup() {
  rm -rf "$working_dir"
}
trap cleanup EXIT

set -eux

# Create a corpus of a single file.
mkdir "$working_dir/corpus"
cat <<EOF > "$working_dir/corpus/file.txt"
kernel void A(global int* a, const int b) {
  if (get_global_id(0) < b) {
    a[get_global_id(0)] *= 2;
  }
}
EOF

# Create a config.
cat <<EOF > "$working_dir/config.pbtxt"
# File: //deeplearning/deepsmith/proto/clgen.proto
# Proto: clgen.Instance
working_dir: "$working_dir/cache"
model {
  corpus {
    local_directory: "$working_dir/corpus"
    ascii_character_atomizer: true
    contentfile_separator: "\n\n"
    preprocessor: "deeplearning.clgen.preprocessors.opencl:Compile"
  }
  architecture {
    backend: TENSORFLOW
    neuron_type: LSTM
    neurons_per_layer: 4
    num_layers: 2
    post_layer_dropout_micros: 0
  }
  training {
    num_epochs: 1
    sequence_length: 4
    batch_size: 2
    shuffle_corpus_contentfiles_between_epochs: false
    adam_optimizer {
      initial_learning_rate_micros: 2000
      learning_rate_decay_per_epoch_micros: 50000
      beta_1_micros: 900000
      beta_2_micros: 999000
      normalized_gradient_clip_micros: 5000000
    }
  }
}
sampler {
  start_text: "kernel void "
  batch_size: 1
  sequence_length: 64
  temperature_micros: 1000000
  termination_criteria {
    maxlen {
      maximum_tokens_in_sample: 10
    }
  }
}
EOF

deeplearning/clgen/clgen \
    --config="$working_dir/config.pbtxt" \
    --min_samples=1
