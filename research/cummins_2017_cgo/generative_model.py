"""This is an implementation of the OpenCL code generator described in:

    ﻿Cummins, C., Petoumenos, P., Zang, W., & Leather, H. (2017). Synthesizing
    Benchmarks for Predictive Modeling. In CGO. IEEE.

Note this is neither the same implementation as was used to generate the data
for the paper, nor the same model hyper-parameters. For that, see the paper's
artifact in //docs/2017_02_cgo/code.
"""
import pathlib
import typing

from absl import app
from absl import flags

from deeplearning.clgen import clgen
from deeplearning.clgen.proto import clgen_pb2
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('clgen_working_dir',
                    str(pathlib.Path('~/.cache/clgen').expanduser()),
                    'The directory for CLgen working files.')

# Corpus options.
flags.DEFINE_string('clgen_corpus_dir',
                    "/mnt/cc/data/datasets/github/corpuses/opencl",
                    "Directory where the corpus is stored.")
flags.DEFINE_boolean('clgen_multichar_tokenizer', False,
                     'If true, use multichar OpenCL token.')

# Model options.
flags.DEFINE_integer('clgen_layer_size', 512, 'Size of LSTM model layers.')
flags.DEFINE_integer('clgen_num_layers', 2, 'Number of layers in LSTM model.')
flags.DEFINE_integer('clgen_max_sample_length', 20000,
                     'The maximum length of CLgen samples.')

# Training options.
flags.DEFINE_integer("clgen_num_epochs", 50, "The number of training epochs.")
flags.DEFINE_integer("clgen_sequence_length", 64, "CLgen sequence length.")
flags.DEFINE_integer("clgen_training_batch_size", 64,
                     "CLgen sampling batch size.")

# Sampling options.
flags.DEFINE_string("clgen_seed_text", "kernel void ",
                    "CLgen sample seed text.")
flags.DEFINE_float("clgen_sample_temperature", 1.0,
                   "CLgen sampling temperature.")
flags.DEFINE_integer("clgen_sample_batch_size", 64,
                     "CLgen sampling btach size.")
flags.DEFINE_integer("clgen_min_sample_count", 0,
                     "If not zero, set the maximum number of samples.")


def CreateCorpusProtoFromFlags() -> corpus_pb2.Corpus:
  corpus = corpus_pb2.Corpus(
      local_directory=FLAGS.clgen_corpus_dir,
      preprocessor=[
          "deeplearning.clgen.preprocessors.opencl:ClangPreprocessWithShim",
          "deeplearning.clgen.preprocessors.opencl:Compile",
          "deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers",
          "deeplearning.clgen.preprocessors.opencl:StripDoubleUnderscorePrefixes",
          "deeplearning.clgen.preprocessors.common:StripDuplicateEmptyLines",
          "deeplearning.clgen.preprocessors.opencl:SanitizeKernelPrototype",
          "deeplearning.clgen.preprocessors.common:StripTrailingWhitespace",
          "deeplearning.clgen.preprocessors.opencl:ClangFormat",
          "deeplearning.clgen.preprocessors.common:MinimumLineCount3",
          "deeplearning.clgen.preprocessors.opencl:Compile",
      ],
      contentfile_separator='\n\n',
  )
  if FLAGS.clgen_multichar_tokenizer:
    corpus.greedy_multichar_atomizer = corpus_pb2.GreedyMulticharAtomizer(
        tokens=[
            "  ",
            "__assert",
            "__attribute",
            "__builtin_astype",
            "__clc_fabs",
            "__clc_fma",
            "__inline",
            "abs",
            "alignas",
            "alignof",
            "atomic_add",
            "auto",
            "barrier",
            "bool",
            "break",
            "case",
            "char",
            "clamp",
            "complex",
            "const",
            "constant",
            "continue",
            "default",
            "defined",
            "do",
            "double",
            "else",
            "enum",
            "error",
            "event_t",
            "extern",
            "fabs",
            "false",
            "float",
            "for",
            "get_global_id",
            "get_global_size",
            "get_local_id",
            "get_local_size",
            "get_num_groups",
            "global",
            "goto",
            "half",
            "if",
            "image1d_array_t",
            "image1d_buffer_t",
            "image1d_t",
            "image2d_array_t",
            "image2d_t",
            "image3d_t",
            "imaginary",
            "include",
            "inline",
            "int",
            "into",
            "kernel",
            "line",
            "local",
            "long",
            "noreturn",
            "pragma",
            "private",
            "quad",
            "read_only",
            "read_write",
            "register",
            "restrict",
            "return",
            "sampler_t",
            "short",
            "shuffle",
            "signed",
            "size_t",
            "sizeof",
            "sqrt",
            "static",
            "struct",
            "switch",
            "true",
            "typedef",
            "u32",
            "uchar",
            "uint",
            "ulong",
            "undef",
            "union",
            "unsigned",
            "void",
            "volatile",
            "while",
            "wide",
            "write_only",
        ])
  else:
    corpus.ascii_character_atomizer = True

  return corpus


def CreateModelProtoFromFlags() -> model_pb2.Model:
  return model_pb2.Model(
      corpus=CreateCorpusProtoFromFlags(),
      architecture=model_pb2.NetworkArchitecture(
          backend=model_pb2.NetworkArchitecture.TENSORFLOW,
          neuron_type=model_pb2.NetworkArchitecture.LSTM,
          neurons_per_layer=FLAGS.clgen_layer_size,
          num_layers=FLAGS.clgen_num_layers,
          post_layer_dropout_micros=0,
      ),
      training=model_pb2.TrainingOptions(
          num_epochs=FLAGS.clgen_num_epochs,
          sequence_length=FLAGS.clgen_sequence_length,
          batch_size=FLAGS.clgen_training_batch_size,
          shuffle_corpus_contentfiles_between_epochs=True,
          adam_optimizer=model_pb2.AdamOptimizer(
              initial_learning_rate_micros=2000,
              learning_rate_decay_per_epoch_micros=50000,
              beta_1_micros=900000,
              beta_2_micros=999000,
              normalized_gradient_clip_micros=5000000,
          ),
      ))


def CreateSamplerProtoFromFlags() -> sampler_pb2.Sampler:
  return sampler_pb2.Sampler(
      start_text=FLAGS.clgen_seed_text,
      batch_size=FLAGS.clgen_sample_batch_size,
      temperature_micros=int(FLAGS.clgen_sample_temperature * 1000000),
      termination_criteria=[
          sampler_pb2.SampleTerminationCriterion(
              symtok=sampler_pb2.SymmetricalTokenDepth(
                  depth_increase_token="{",
                  depth_decrease_token="}",
              )),
          sampler_pb2.SampleTerminationCriterion(
              maxlen=sampler_pb2.MaxTokenLength(
                  maximum_tokens_in_sample=FLAGS.clgen_max_sample_length,)),
      ],
  )


def CreateInstanceProtoFromFlags() -> clgen_pb2.Instance:
  return clgen_pb2.Instance(
      working_dir=FLAGS.clgen_working_dir,
      model=CreateModelProtoFromFlags(),
      sampler=CreateSamplerProtoFromFlags(),
  )


def CreateInstanceFromFlags() -> clgen.Instance:
  return clgen.Instance(CreateInstanceProtoFromFlags())


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  instance = CreateInstanceFromFlags()
  sample_count = 0
  with instance.Session():
    instance.Train()
    while (FLAGS.clgen_min_sample_count and
           sample_count < FLAGS.clgen_min_sample_count):
      sample_count += len(
          instance.Sample(min_num_samples=FLAGS.clgen_min_sample_count))


if __name__ == '__main__':
  app.run(main)
