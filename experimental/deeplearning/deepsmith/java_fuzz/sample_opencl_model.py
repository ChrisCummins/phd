"""Debugging script which uses the same training and sampling logic as
:sample_java_model, but using an OpenCL corpus.

Example usage:

  $ bazel run //experimental/deeplearning/deepsmith/java_fuzz:sample_opencl_model -- \
      --java_clgen_working_dir=/var/phd/experimental/deeplearning/deepsmith/java_fuzz/opencl_clgen_cache \
      --samples_db='file:///var/phd/db/cc1.mysql?clgen_opencl_samples_2019.07.29?charset=utf8' \
      --neurons_per_layer=512 \
      --clgen_corpus_dir='/var/phd/datasets/github/corpuses/opencl' \
      --clgen_multichar_tokenizer
"""
from deeplearning.clgen import clgen
from deeplearning.clgen.proto import corpus_pb2
from experimental.deeplearning.deepsmith.java_fuzz import sample_java_model as java
from research.cummins_2017_cgo import generative_model as opencl
from labm8 import app

FLAGS = app.FLAGS

app.DEFINE_boolean(
    'use_encoded_contentfiles_db', False,
    'If set, use the --java_encoded_contentfiles flag to as the training '
    'corpus.')


def main():
  """Main entry point."""
  config = java.MakeClgenInstanceConfig(
      FLAGS.java_clgen_working_dir,
      FLAGS.java_encoded_contentfiles(),
      FLAGS.java_training_epochs,
      'kernel void A(',  # OpenCL-specific seed text.
      FLAGS.neurons_per_layer,
      FLAGS.num_layers)
  if not FLAGS.use_encoded_contentfiles_db:
    # Replace the Java corpus with an OpenCL one.
    config.model.corpus.CopyFrom(opencl.CreateCorpusProtoFromFlags())
  samples_db = FLAGS.samples_db()
  java.TrainAndSampleInstance(clgen.Instance(config), samples_db)


if __name__ == '__main__':
  app.Run(main)
