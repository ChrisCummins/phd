"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
import pathlib
import typing

from deeplearning.clgen import clgen
from deeplearning.clgen import sample_observers
from deeplearning.clgen.proto import clgen_pb2
from labm8.py import app
from labm8.py import pbutil
from labm8.py import prof

FLAGS = app.FLAGS

# The CLgen instance to benchmark. This is a multichar token model with a 512x2
# TensorFlow LSTM.
INSTANCE_TO_BENCHMARK = pbutil.FromString(
  """
# File: //deeplearning/clgen/proto/clgen.proto
# Proto: clgen.Instance
working_dir: "/tmp/phd/deeplearning/clgen/working_dir"
model {
  corpus {
    local_directory: "$HOME/data/datasets/github/corpuses/opencl"
    greedy_multichar_atomizer {
      tokens: "  "
      tokens: "__assert"
      tokens: "__attribute"
      tokens: "__builtin_astype"
      tokens: "__clc_fabs"
      tokens: "__clc_fma"
      tokens: "__inline"
      tokens: "abs"
      tokens: "alignas"
      tokens: "alignof"
      tokens: "atomic_add"
      tokens: "auto"
      tokens: "barrier"
      tokens: "bool"
      tokens: "break"
      tokens: "case"
      tokens: "char"
      tokens: "clamp"
      tokens: "complex"
      tokens: "const"
      tokens: "constant"
      tokens: "continue"
      tokens: "default"
      tokens: "defined"
      tokens: "do"
      tokens: "double"
      tokens: "else"
      tokens: "enum"
      tokens: "error"
      tokens: "event_t"
      tokens: "extern"
      tokens: "fabs"
      tokens: "false"
      tokens: "float"
      tokens: "for"
      tokens: "get_global_id"
      tokens: "get_global_size"
      tokens: "get_local_id"
      tokens: "get_local_size"
      tokens: "get_num_groups"
      tokens: "global"
      tokens: "goto"
      tokens: "half"
      tokens: "if"
      tokens: "image1d_array_t"
      tokens: "image1d_buffer_t"
      tokens: "image1d_t"
      tokens: "image2d_array_t"
      tokens: "image2d_t"
      tokens: "image3d_t"
      tokens: "imaginary"
      tokens: "include"
      tokens: "inline"
      tokens: "int"
      tokens: "into"
      tokens: "kernel"
      tokens: "line"
      tokens: "local"
      tokens: "long"
      tokens: "noreturn"
      tokens: "pragma"
      tokens: "private"
      tokens: "quad"
      tokens: "read_only"
      tokens: "read_write"
      tokens: "register"
      tokens: "restrict"
      tokens: "return"
      tokens: "sampler_t"
      tokens: "short"
      tokens: "shuffle"
      tokens: "signed"
      tokens: "size_t"
      tokens: "sizeof"
      tokens: "sqrt"
      tokens: "static"
      tokens: "struct"
      tokens: "switch"
      tokens: "true"
      tokens: "typedef"
      tokens: "u32"
      tokens: "uchar"
      tokens: "uint"
      tokens: "ulong"
      tokens: "undef"
      tokens: "union"
      tokens: "unsigned"
      tokens: "void"
      tokens: "volatile"
      tokens: "while"
      tokens: "wide"
      tokens: "write_only"
    }
    contentfile_separator: "\\n\\n"
    preprocessor: "deeplearning.clgen.preprocessors.opencl:ClangPreprocessWithShim"
    preprocessor: "deeplearning.clgen.preprocessors.opencl:Compile"
    preprocessor: "deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers"
    preprocessor: "deeplearning.clgen.preprocessors.opencl:StripDoubleUnderscorePrefixes"
    preprocessor: "deeplearning.clgen.preprocessors.common:StripDuplicateEmptyLines"
    preprocessor: "deeplearning.clgen.preprocessors.opencl:SanitizeKernelPrototype"
    preprocessor: "deeplearning.clgen.preprocessors.common:StripTrailingWhitespace"
    preprocessor: "deeplearning.clgen.preprocessors.opencl:ClangFormat"
    preprocessor: "deeplearning.clgen.preprocessors.common:MinimumLineCount3"
    # TODO(cec): Temporarily disabling this: preprocessor: "deeplearning.clgen.preprocessors.opencl:Compile"
  }
  architecture {
    backend: TENSORFLOW
    neuron_type: LSTM
    neurons_per_layer: 512
    num_layers: 2
    post_layer_dropout_micros: 0
  }
  training {
    num_epochs: 50
    sequence_length: 64
    batch_size: 64
    shuffle_corpus_contentfiles_between_epochs: true
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
  batch_size: 64
  temperature_micros: 1000000  # = 1.0 real value
  termination_criteria {
    symtok {
      depth_increase_token: "{"
      depth_decrease_token: "}"
    }
  }
  termination_criteria {
    maxlen {
      maximum_tokens_in_sample: 5000
    }
  }
}
""",
  clgen_pb2.Instance(),
)

# The number of samples to take from the trained model.
NUM_SAMPLES = 5000

# Path to write benchmark log to.
LOG_PATH = pathlib.Path("/tmp/phd/deeplearning/clgen/benchmark.txt")


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(" ".join(argv[1:])))

  LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

  with open(LOG_PATH, "w") as f:
    print("Benchmark of instance:\n{INSTANCE_TO_BENCHMARK}", file=f)

    with prof.ProfileToFile(f, "instance"):
      instance = clgen.Instance(INSTANCE_TO_BENCHMARK)
    with prof.ProfileToFile(f, "training"):
      instance.Train()
    with prof.ProfileToFile(f, f"{NUM_SAMPLES} samples"):
      instance.Sample(
        sample_observers=[sample_observers.MaxSampleCountObserver(NUM_SAMPLES)]
      )


if __name__ == "__main__":
  app.RunWithArgs(main)
