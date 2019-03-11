# NCC: Neural Code Comprehension
# https://github.com/spcl/ncc
# Copyright 2018 ETH Zurich
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
"""Training workflow for optimal device mapping prediction

Output when reproducing published results:

$ bazel run //deeplearning/ncc:train_task_devmap -- --v=1 \
    --embeddings_file=$PHD/deeplearning/ncc/published_results/emb.p \
    --vocabulary_zip_path=$PHD/deeplearning/ncc/published_results/vocabulary.zip

# ... snip ...

--- Prediction results
                                 Correct?    Speedup
Platform        Benchmark Suite
AMD Tahiti 7970 AMD SDK          0.500000   0.860580
                NPB              0.834915   2.926740
                NVIDIA SDK       0.583333   2.158376
                Parboil          0.894737   6.350277
                Polybench        0.740741  13.994454
                Rodinia          0.645161   3.781416
                SHOC             0.729167   1.006453
NVIDIA GTX 970  AMD SDK          0.500000   0.781197
                NPB              0.853890   1.417215
                NVIDIA SDK       0.666667   1.151123
                Parboil          0.789474   1.397015
                Polybench        0.777778   1.155757
                Rodinia          0.516129   1.082831
                SHOC             0.770833   1.877359
--- Prediction results (summarized)
                 Correct?   Speedup
Platform
AMD Tahiti 7970  0.804412  3.303089
NVIDIA GTX 970   0.816176  1.403845
--- Model comparison: prediction accuracy
                 Static mapping  Grewe et al.   DeepTune  DeepTuneInst2Vec
AMD Tahiti 7970       58.823529     73.382353  83.676471         80.441176
NVIDIA GTX 970        56.911765     72.941176  80.294118         81.617647
Average               57.867647     73.161765  81.985294         81.029412
--- Model comparison: speedups
                 Static mapping  Grewe et al.  DeepTune  DeepTuneInst2Vec
AMD Tahiti 7970             1.0      2.905822  3.335612          3.303089
NVIDIA GTX 970              1.0      1.264801  1.412222          1.403845
Average                     1.0      2.085312  2.373917          2.353467
"""
import os
import pathlib
import pickle

import numpy as np
import pandas as pd

from deeplearning.ncc import task_utils
from deeplearning.ncc import vocabulary
from labm8 import app
from labm8 import bazelutil
from labm8 import fs

# Parameters of devmap
app.DEFINE_string('input_data', '/tmp/phd/deeplearning/ncc/task/devmap',
                  'Path to input data')
app.DEFINE_string(
    'out', '/tmp/phd/deeplearning/ncc/task/devmap',
    'Path to folder in which to write saved Keras models and predictions')
app.DEFINE_string(
    'vocabulary_zip_path', None,
    'Path to the vocabulary zip file associated with those embeddings')
app.DEFINE_string('device', 'all',
                  'Device to evaluate model on. Options: all, amd, nvidia')
app.DEFINE_integer('num_epochs', 50, 'number of training epochs')
app.DEFINE_integer('batch_size', 64, 'training batch size')
app.DEFINE_integer('dense_layer', 32, 'dense layer size')
app.DEFINE_boolean('print_summary', False, 'Print summary of Keras model')

FLAGS = app.FLAGS


def platform2str(platform: str) -> str:
  if platform == "amd":
    return "AMD Tahiti 7970"
  elif platform == "nvidia":
    return "NVIDIA GTX 970"
  else:
    raise LookupError


def escape_suite_name(g: str) -> str:
  c = g.split('-')
  if c[0] == "amd" or c[0] == "nvidia":
    return c[0].upper() + " SDK"
  if c[0] == "npb" or c[0] == "shoc":
    return c[0].upper()
  elif c[0] == "parboil" or c[0] == "polybench" or c[0] == "rodinia":
    return c[0].capitalize()
  else:
    raise LookupError


def escape_benchmark_name(g: str) -> str:
  c = g.split('-')
  return escape_suite_name(c[0]).split()[0] + "." + c[-2]


def auxiliary_inputs(df: pd.DataFrame) -> np.array:
  return np.array([
      df["transfer"].values,
      df["wgsize"].values,
  ]).T


def encode_1hot(y: np.array) -> np.array:
  labels = np.vstack([np.expand_dims(x, axis=0) for x in y])
  l2 = [x[0] for x in labels]
  l1 = [not x for x in l2]
  return np.array(list(zip(l1, l2)), dtype=np.int32)


# TODO(cec): Actually encode the srcs, don't read CSV files.
def encode_srcs(data_folder, df: pd.DataFrame) -> np.array:
  from keras.preprocessing.sequence import pad_sequences

  # Get the 'unknown' vocab index.
  with vocabulary.VocabularyZipFile(FLAGS.vocabulary_zip_path) as vocab:
    unk_index = vocab.unknown_token_index

  # Get list of source file names
  data_folder = os.path.join(data_folder, 'kernels_seq')
  input_files = df["benchmark"].values  # list of strings of benchmark names
  dataset = df["dataset"].values  # list of strings of dataset descriptions
  num_files = len(input_files)
  num_unks = 0
  seq_lengths = list()

  app.Log(1, 'Preparing to read %d input files from folder %s', num_files,
          data_folder)
  seqs = list()
  for i in range(num_files):
    file = input_files[i]
    dat = dataset[i]
    if file[:3] == "npb":
      # concatenate data set size
      file += '_' + str(dat)
    file = os.path.join(data_folder, file + '_seq.csv')
    if os.path.exists(file):
      # load sequence
      with open(file, 'r') as f:
        seq = f.read().splitlines()
        assert len(seq) > 0, 'Found empty file: ' + file
      num_unks += seq.count(str(unk_index))
      seq_lengths.append(len(seq))
      seqs.append([int(s) for s in seq])
    else:
      assert True, 'input file not found: ' + file

  max_len = max(seq_lengths)
  app.Log(1, 'Sequence lengths: min=%d, avg=%.2f, max=%d', min(seq_lengths),
          np.mean(seq_lengths), max_len)
  app.Log(1, 'Number of \'UNK\': %d', num_unks)
  app.Log(1, 'Percentage of \'UNK\': %.3f %% among all stmts',
          (num_unks * 100) / sum(seq_lengths))
  app.Log(1, '\'UNK\' index: %d', unk_index)

  encoded = np.array(pad_sequences(seqs, maxlen=max_len, value=unk_index))
  return np.vstack([np.expand_dims(x, axis=0) for x in encoded]), max_len


# TODO(cec): Code duplication with
# //deeplearning/deeptune/opencl/heterogeneous_mapping:models.
class NCC_devmap:
  __name__ = "NCC_devmap"
  __basename__ = "ncc_devmap"

  def init(self, seed: int, maxlen: int, embedding_dim: int,
           dense_layer_size: int):
    from keras.layers import Input, LSTM, Dense
    from keras.layers.merge import Concatenate
    from keras.layers.normalization import BatchNormalization
    from keras.models import Model

    np.random.seed(seed)

    # Keras model
    inp = Input(
        shape=(
            maxlen,
            embedding_dim,
        ), dtype="float32", name="code_in")
    x = LSTM(
        embedding_dim, implementation=1, return_sequences=True,
        name="lstm_1")(inp)
    x = LSTM(embedding_dim, implementation=1, name="lstm_2")(x)
    langmodel_out = Dense(2, activation="sigmoid")(x)

    # Auxiliary inputs. wgsize and dsize.
    auxiliary_inputs = Input(shape=(2,))
    x = Concatenate()([auxiliary_inputs, x])
    x = BatchNormalization()(x)
    x = Dense(dense_layer_size, activation="relu")(x)
    out = Dense(2, activation="sigmoid")(x)

    self.model = Model(
        inputs=[auxiliary_inputs, inp], outputs=[out, langmodel_out])
    self.model.compile(
        optimizer="adam",
        metrics=['accuracy'],
        loss=["categorical_crossentropy", "categorical_crossentropy"],
        loss_weights=[1., .2])
    app.Log(1, 'Built Keras model')

    return self

  def save(self, outpath):
    self.model.save(outpath)

  def restore(self, inpath):
    from keras.models import load_model
    self.model = load_model(inpath)

  def train(self, epochs: int, batch_size: int, **train) -> None:
    self.model.fit([train["aux_in"], train["sequences"]],
                   [train["y_1hot"], train["y_1hot"]],
                   epochs=epochs,
                   batch_size=batch_size,
                   verbose=train["verbose"],
                   shuffle=True)

  def predict(self, batch_size, **test):
    p = np.array(
        self.model.predict([test["aux_in"], test["sequences"]],
                           batch_size=batch_size,
                           verbose=test["verbose"]))
    indices = [np.argmax(x) for x in p[0]]
    return indices


########################################################################################################################
# Evaluate
########################################################################################################################
# Set seed for reproductibility
seed = 204


def evaluate(model, device, data_folder, out_folder, embeddings,
             dense_layer_size, print_summary, num_epochs,
             batch_size) -> pd.DataFrame:
  from sklearn.model_selection import StratifiedKFold

  # Create device list
  if device == 'all':
    device_list = ["amd", "nvidia"]
  else:
    device_list = [device]

  data = []
  for i, platform in enumerate(device_list):
    platform_name = platform2str(platform)

    # Load runtime data
    data_file = os.path.join(data_folder, "cgo17-{}.csv".format(platform))
    print('\n--- Read data from', data_file)
    df = pd.read_csv(data_file)

    # Encode input source codes
    sequences, maxlen = encode_srcs(data_folder, df)

    # Load embeddings
    import tensorflow as tf  # for embeddings lookup
    embedding_matrix_normalized = tf.nn.l2_normalize(embeddings, axis=1)
    vocabulary_size, embedding_dimension = embedding_matrix_normalized.shape
    seq_ = tf.placeholder(dtype=tf.int32)
    # Tensor of shape (num_input_files, sequence length, embbedding dimension)
    embedding_input_ = tf.nn.embedding_lookup(embedding_matrix_normalized, seq_)
    with tf.Session() as sess:
      embedding_input = sess.run(embedding_input_, feed_dict={seq_: sequences})

    # Values used for training & predictions
    aux_in = auxiliary_inputs(df)

    # Optimal mappings
    y = np.array([1 if x == "GPU" else 0 for x in df["oracle"].values])
    y_1hot = encode_1hot(y)

    # 10-fold cross-validation
    n_splits = 10
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for j, (train_index, test_index) in enumerate(kf.split(sequences, y)):
      print('--- Cross validation step [', j, '/ ', n_splits, ']')

      model_basename = model.__basename__
      model_path = os.path.join(
          out_folder, "models/{model_basename}-{platform}-{j}.model".format(
              model_basename=model_basename, platform=platform, j=j))
      predictions_path = os.path.join(
          out_folder,
          "predictions/{model_basename}-{platform}-{j}.result".format(
              model_basename=model_basename, platform=platform, j=j))

      if fs.exists(predictions_path):
        # load result from cache
        print("\tFound predictions in", predictions_path, ", skipping...")
        with open(predictions_path, 'rb') as infile:
          p = pickle.load(infile)
      else:

        if fs.exists(model_path):
          # restore trained model from cache
          print("\n\tFound trained model in", model_path, ", skipping...")
          model.restore(model_path)
        else:

          # Initialize model and print summary
          model.init(
              seed=seed,
              maxlen=maxlen,
              embedding_dim=int(embedding_dimension),
              dense_layer_size=dense_layer_size)
          if print_summary:
            model.model.summary()

          # Train and cache a model
          print('\n--- Training model... ')
          model.train(
              df=df,
              aux_in=aux_in[train_index],
              sequences=embedding_input[train_index, :, :],
              y=y[train_index],
              y_1hot=y_1hot[train_index],
              verbose=False,
              epochs=num_epochs,
              batch_size=batch_size)
          fs.mkdir(fs.dirname(model_path))
          model.save(model_path)
          print('\tsaved model to', model_path)

        # test model
        print('\n--- Testing model... ')
        p = model.predict(
            batch_size=batch_size,
            aux_in=aux_in[test_index],
            sequences=embedding_input[test_index, :, :],
            y=y[test_index],
            y_1hot=y_1hot[test_index],
            verbose=False)

        # cache results
        fs.mkdir(fs.dirname(predictions_path))
        with open(predictions_path, 'wb') as outfile:
          pickle.dump(p, outfile)
        print('\tsaved predictions to', predictions_path)

      benchmarks = df['benchmark'].values[test_index]  # benchmarks names
      o = y[test_index]  # oracle device mappings (true values)
      correct = p == o  # predictions' correctness
      # runtimes of baseline mapping (CPU on AMD, GPU on NVIDIA)
      zero_r_dev = "runtime_cpu" if platform == "amd" else "runtime_gpu"
      zer_r_runtimes = df[zero_r_dev][test_index]
      # speedups of predictions
      runtimes = df[['runtime_cpu', 'runtime_gpu']].values[test_index]
      p_runtimes = [r[p_] for p_, r in zip(p, runtimes)]
      p_speedup = zer_r_runtimes / p_runtimes

      # sanity check
      assert (len(benchmarks) == len(o) == len(correct) == len(p) ==
              len(p_speedup))

      # record results
      for benchmark_, o_, p_, correct_, p_speedup_ in zip(
          benchmarks, o, p, correct, p_speedup):
        data.append({
            "Model": model_basename,
            "Platform": platform_name,
            'Benchmark': escape_benchmark_name(benchmark_),
            'Benchmark Suite': escape_suite_name(benchmark_),
            "Oracle Mapping": o_,
            "Predicted Mapping": p_,
            "Correct?": correct_,
            "Speedup": p_speedup_,
        })

  return pd.DataFrame(
      data,
      index=range(1,
                  len(data) + 1),
      columns=[
          "Model", "Platform", "Benchmark", "Benchmark Suite", "Oracle Mapping",
          "Predicted Mapping", "Correct?", "Speedup"
      ])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')

  # Don't truncate output when printing pandas tables.
  pd.set_option('display.max_columns', None)
  pd.set_option('display.max_rows', None)

  # Setup
  # Get flag values
  embeddings = task_utils.ReadEmbeddingFileFromFlags()
  out = FLAGS.out
  if not os.path.exists(out):
    os.makedirs(out)
  device = FLAGS.device
  assert device in ['all', 'amd', 'nvidia'], \
    'Choose device among: all, amd, nvidia'
  dense_layer_size = FLAGS.dense_layer
  print_summary = FLAGS.print_summary
  num_epochs = FLAGS.num_epochs
  batch_size = FLAGS.batch_size
  input_data = FLAGS.input_data

  # Unpack data archive if necessary.
  if not os.path.exists(os.path.join(input_data, 'kernels_ir')):
    dataset = bazelutil.DataArchive(
        'phd/deeplearning/ncc/published_results/task_devmap.zip')
    dataset.ExtractAll(pathlib.Path(input_data))

  with vocabulary.VocabularyZipFile(FLAGS.vocabulary_zip_path) as vocab:
    task_utils.CreateSeqDirFromIr(os.path.join(input_data, 'kernels_ir'), vocab)

  # Reference values copied from:
  # https://github.com/ChrisCummins/paper-end2end-dl/blob/master/code/Case%20Study%20A.ipynb
  static_pred_vals = [58.823529, 56.911765]
  static_pred_mean = 57.867647
  static_sp_vals = [1.0, 1.0]
  static_sp_mean = 1.0
  grewe_pred_vals = [73.382353, 72.941176]
  grewe_pred_mean = 73.161765
  grewe_sp_vals = [2.905822, 1.264801]
  grewe_sp_mean = 2.085312
  deeptune_pred_vals = [83.676471, 80.294118]
  deeptune_pred_mean = 81.985294
  deeptune_sp_vals = [3.335612, 1.412222]
  deeptune_sp_mean = 2.373917

  # Train model
  app.Log(1, "Evaluating ncc model")
  ncc_devmap = evaluate(NCC_devmap(), device, input_data, out, embeddings,
                        dense_layer_size, print_summary, num_epochs, batch_size)

  # Print results
  print('--- Prediction results')
  print(
      ncc_devmap.groupby(
          ['Platform',
           'Benchmark Suite'])['Platform', 'Correct?', 'Speedup'].mean())
  print('--- Prediction results (summarized)')
  print(
      ncc_devmap.groupby(
          ['Platform'])['Platform', 'Correct?', 'Speedup'].mean())

  # Model comparison: prediction accuracy
  print('--- Model comparison: prediction accuracy')
  d = list()
  d.append(np.append(static_pred_vals, static_pred_mean))
  d.append(np.append(grewe_pred_vals, grewe_pred_mean))
  d.append(np.append(deeptune_pred_vals, deeptune_pred_mean))
  d.append(
      np.append(
          ncc_devmap.groupby(['Platform'])['Correct?'].mean().values * 100,
          ncc_devmap['Correct?'].mean() * 100))
  d = np.array(d).T.reshape(3, 4)
  print(
      pd.DataFrame(
          d,
          columns=[
              'Static mapping', 'Grewe et al.', 'DeepTune', 'DeepTuneInst2Vec'
          ],
          index=['AMD Tahiti 7970', 'NVIDIA GTX 970', 'Average']))

  # Model comparison: speedups
  print('--- Model comparison: speedups')
  d = list()
  d.append(np.append(static_sp_vals, static_sp_mean))
  d.append(np.append(grewe_sp_vals, grewe_sp_mean))
  d.append(np.append(deeptune_sp_vals, deeptune_sp_mean))
  d.append(
      np.append(
          ncc_devmap.groupby(['Platform'])['Speedup'].mean().values,
          ncc_devmap['Speedup'].mean()))
  d = np.array(d).T.reshape(3, 4)
  print(
      pd.DataFrame(
          d,
          columns=[
              'Static mapping', 'Grewe et al.', 'DeepTune', 'DeepTuneInst2Vec'
          ],
          index=['AMD Tahiti 7970', 'NVIDIA GTX 970', 'Average']))
  app.Log(1, 'done')


if __name__ == '__main__':
  app.RunWithArgs(main)
