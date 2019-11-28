"""Utilities for LSTM models."""
import keras
import tensorflow as tf
from keras.backend import tensorflow_backend

from deeplearning.ml4pl.models import classifier_base
from labm8.py import app

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


def SetAllowedGrowthOnKerasSession():
  """Allow growth on GPU."""
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tensorflow_backend.set_session(sess)
