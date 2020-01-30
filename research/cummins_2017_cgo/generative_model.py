# Copyright 2017-2020 Chris Cummins <chrisc.101@gmail.com>.
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
"""This is an implementation of the OpenCL code generator described in:

    ﻿Cummins, C., Petoumenos, P., Zang, W., & Leather, H. (2017). Synthesizing
    Benchmarks for Predictive Modeling. In CGO. IEEE.

Note this is neither the same implementation as was used to generate the data
for the paper, nor the same model hyper-parameters. For that, see the paper's
artifact in //docs/2017_02_cgo/code.
"""
import json
import pathlib
import typing

from deeplearning.clgen import clgen
from deeplearning.clgen import sample_observers
from deeplearning.clgen.proto import clgen_pb2
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import pbutil

FLAGS = app.FLAGS

TOKEN_LISTS = json.loads(
  bazelutil.DataString("phd/deeplearning/clgen/corpuses/token_lists.json")
)

app.DEFINE_string(
  "clgen_instance",
  None,
  "Path to a clgen.Instance proto file containing a full "
  "CLgen configuration.",
)

app.DEFINE_string(
  "clgen_working_dir",
  str(pathlib.Path("~/.cache/clgen").expanduser()),
  "The directory for CLgen working files.",
)

# Corpus options.
app.DEFINE_string(
  "clgen_corpus_dir",
  "/mnt/cc/data/datasets/github/corpuses/opencl",
  "Directory where the corpus is stored.",
)
app.DEFINE_boolean(
  "clgen_multichar_tokenizer", False, "If true, use multichar OpenCL token."
)

# Model options.
app.DEFINE_integer("clgen_layer_size", 512, "Size of LSTM model layers.")
app.DEFINE_integer("clgen_num_layers", 2, "Number of layers in LSTM model.")
app.DEFINE_integer(
  "clgen_max_sample_length",
  20000,
  "The maximum length of CLgen samples. If 0, no limit.",
)

# Training options.
app.DEFINE_integer("clgen_num_epochs", 50, "The number of training epochs.")
app.DEFINE_integer(
  "clgen_training_sequence_length", 64, "CLgen training sequence length."
)
app.DEFINE_integer(
  "clgen_training_batch_size", 64, "CLgen training batch size."
)

# Sampling options.
app.DEFINE_string("clgen_seed_text", "kernel void ", "CLgen sample seed text.")
app.DEFINE_float("clgen_sample_temperature", 1.0, "CLgen sampling temperature.")
app.DEFINE_integer(
  "clgen_sample_sequence_length", 1024, "CLgen sampling sequence length."
)
app.DEFINE_integer("clgen_sample_batch_size", 64, "CLgen sampling batch size.")

# Sample observer options.
app.DEFINE_integer(
  "clgen_min_sample_count", 0, "If not zero, set the maximum number of samples."
)
app.DEFINE_boolean(
  "clgen_cache_sample_protos",
  False,
  "If set, save generated sample protos in the CLgen cache.",
)
app.DEFINE_boolean(
  "clgen_print_samples", True, "If set, print CLgen sample outputs."
)
app.DEFINE_list(
  "clgen_preprocessor",
  [
    "deeplearning.clgen.preprocessors.opencl:ClangPreprocessWithShim",
    "deeplearning.clgen.preprocessors.opencl:Compile",
    "deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers",
    "deeplearning.clgen.preprocessors.opencl:StripDoubleUnderscorePrefixes",
    "deeplearning.clgen.preprocessors.common:StripDuplicateEmptyLines",
    "deeplearning.clgen.preprocessors.opencl:SanitizeKernelPrototype",
    "deeplearning.clgen.preprocessors.common:StripTrailingWhitespace",
    "deeplearning.clgen.preprocessors.opencl:ClangFormat",
    "deeplearning.clgen.preprocessors.common:MinimumLineCount3",
    "deeplearning.clgen.preprocessors.opencl:StripDoubleUnderscorePrefixes",
    "deeplearning.clgen.preprocessors.opencl:Compile",
  ],
  "A list of pre-processors to run on the corpus.",
)


def CreateCorpusProtoFromFlags() -> corpus_pb2.Corpus:
  corpus = corpus_pb2.Corpus(
    local_directory=FLAGS.clgen_corpus_dir,
    preprocessor=FLAGS.clgen_preprocessor,
    contentfile_separator="\n\n",
  )
  if FLAGS.clgen_multichar_tokenizer:
    corpus.greedy_multichar_atomizer.CopyFrom(
      corpus_pb2.GreedyMulticharAtomizer(tokens=TOKEN_LISTS["opencl"]["tokens"])
    )
  else:
    corpus.ascii_character_atomizer = True

  return corpus


def CreateModelProtoFromFlags(corpus: corpus_pb2.Corpus) -> model_pb2.Model:
  return model_pb2.Model(
    corpus=corpus,
    architecture=model_pb2.NetworkArchitecture(
      backend=model_pb2.NetworkArchitecture.TENSORFLOW,
      neuron_type=model_pb2.NetworkArchitecture.LSTM,
      neurons_per_layer=FLAGS.clgen_layer_size,
      num_layers=FLAGS.clgen_num_layers,
      post_layer_dropout_micros=0,
    ),
    training=model_pb2.TrainingOptions(
      num_epochs=FLAGS.clgen_num_epochs,
      sequence_length=FLAGS.clgen_training_sequence_length,
      batch_size=FLAGS.clgen_training_batch_size,
      shuffle_corpus_contentfiles_between_epochs=True,
      adam_optimizer=model_pb2.AdamOptimizer(
        initial_learning_rate_micros=2000,
        learning_rate_decay_per_epoch_micros=50000,
        beta_1_micros=900000,
        beta_2_micros=999000,
        normalized_gradient_clip_micros=5000000,
      ),
    ),
  )


def CreateSamplerProtoFromFlags() -> sampler_pb2.Sampler:
  sampler = sampler_pb2.Sampler(
    start_text=FLAGS.clgen_seed_text,
    batch_size=FLAGS.clgen_sample_batch_size,
    sequence_length=FLAGS.clgen_sample_sequence_length,
    temperature_micros=int(FLAGS.clgen_sample_temperature * 1000000),
    termination_criteria=[
      sampler_pb2.SampleTerminationCriterion(
        symtok=sampler_pb2.SymmetricalTokenDepth(
          depth_increase_token="{", depth_decrease_token="}",
        )
      ),
    ],
  )
  if FLAGS.clgen_max_sample_length:
    sampler.termination_criteria.extend(
      [
        sampler_pb2.SampleTerminationCriterion(
          maxlen=sampler_pb2.MaxTokenLength(
            maximum_tokens_in_sample=FLAGS.clgen_max_sample_length,
          )
        ),
      ]
    )
  return sampler


def CreateInstanceProtoFromFlags() -> clgen_pb2.Instance:
  if FLAGS.clgen_instance:
    return pbutil.FromFile(
      pathlib.Path(FLAGS.clgen_instance), clgen_pb2.Instance()
    )
  else:
    return clgen_pb2.Instance(
      working_dir=FLAGS.clgen_working_dir,
      model=CreateModelProtoFromFlags(CreateCorpusProtoFromFlags()),
      sampler=CreateSamplerProtoFromFlags(),
    )


def CreateInstanceFromFlags() -> clgen.Instance:
  return clgen.Instance(CreateInstanceProtoFromFlags())


def SampleObserversFromFlags() -> typing.List[sample_observers.SampleObserver]:
  """Create sample observers for use with model.Sample() from flags values."""
  observers = []
  if FLAGS.clgen_min_sample_count >= 0:
    app.Warning(
      "--clgen_min_sample_count <= 0 means that sampling (and this "
      "process) will never terminate!"
    )
    observers.append(
      sample_observers.MaxSampleCountObserver(FLAGS.clgen_min_sample_count)
    )
  if FLAGS.clgen_cache_sample_protos:
    observers.append(sample_observers.LegacySampleCacheObserver())
  if FLAGS.clgen_print_samples:
    observers.append(sample_observers.PrintSampleObserver())
  return observers


def main():
  """Main entry point."""
  instance = CreateInstanceFromFlags()
  sample_count = 0
  with instance.Session():
    instance.Train()
    while (
      FLAGS.clgen_min_sample_count
      and sample_count < FLAGS.clgen_min_sample_count
    ):
      sample_count += len(
        instance.Sample(min_num_samples=FLAGS.clgen_min_sample_count)
      )


if __name__ == "__main__":
  app.Run(main)
