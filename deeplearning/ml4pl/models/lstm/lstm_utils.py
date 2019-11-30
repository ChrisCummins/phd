"""Utilities for LSTM models."""
import keras
import tensorflow as tf

from deeplearning.ml4pl.models import classifier_base
from labm8.py import app

app.DEFINE_boolean(
  "cudnn_lstm",
  True,
  "If set, use CuDNNLSTM implementation. Else use default "
  "Keras implementation",
)
# This must be a model flag because the CuDNNLSTM and LSTM implementations are
# not compatible, and do not even compute the same thing.
classifier_base.MODEL_FLAGS.add("cudnn_lstm")

FLAGS = app.FLAGS


def MakeLstm(*args, **kwargs):
  if FLAGS.cudnn_lstm:
    return keras.layers.CuDNNLSTM(*args, **kwargs)
  else:
    return keras.layers.LSTM(*args, **kwargs, implementation=1)


def SetAllowedGrowthOnKerasSession():
  """Allow growth on GPU."""
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.compat.v1.Session(config=config)
  tf.compat.v1.keras.backend.set_session(session)
