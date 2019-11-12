"""Utilities for LSTM models."""
import keras

from deeplearning.ml4pl.models import classifier_base
from labm8 import app

app.DEFINE_boolean(
    'cudnn_lstm', True,
    'If set, use CuDNNLSTM implementation. Else use default '
    'Keras implementation')
# TODO(cec): Are weights of CuDNNLSTM and LSTM compatible? If so, no need for
# this to be a model flag.
classifier_base.MODEL_FLAGS.add("cudnn_lstm")

FLAGS = app.FLAGS


def MakeLstm(*args, **kwargs):
  if FLAGS.cudnn_lstm:
    return keras.layers.CuDNNLSTM(*args, **kwargs)
  else:
    return keras.layers.LSTM(*args, **kwargs, implementation=1)
