import pickle

import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import merge
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.models import Model
from sklearn.tree import DecisionTreeClassifier


def cgo13():
  def create_model(seed=None, **kwargs):
    """ instantiate a model """
    return DecisionTreeClassifier(
      random_state=seed,
      splitter="best",
      max_depth=5,
      min_samples_leaf=5,
      criterion="entropy",
    )

  def train_fn(model, train, *args, seed=None, **kwargs):
    """ train a model """
    np.random.seed(seed)
    model.fit(train["x_4"], train["y"])
    return {}

  def test_fn(model, test, seed, *args, **kwargs):
    """ make predictions for test data """
    np.random.seed(seed)
    return model.predict(test["x_4"])

  def save_fn(outpath, model):
    """ save a trained model """
    with open(outpath, "wb") as outfile:
      pickle.dump(model, outfile)

  def load_fn(inpath):
    """ load a trained model """
    with open(inpath, "rb") as infile:
      model = pickle.load(infile)
    return model

  return {
    "name": "cgo13",
    "create_model": create_model,
    "train_fn": train_fn,
    "test_fn": test_fn,
    "save_fn": save_fn,
    "load_fn": load_fn,
  }


def harry():
  batch_size = 64

  def create_model(*args, data_desc=None, **kwargs):
    """ instantiate a model """
    atomizer = data_desc["atomizer"]
    seq_length = data_desc["seq_length"]

    embedding_vector_length = 64
    vocab_size = atomizer.vocab_size + 1

    dyn_inputs = Input(shape=(2,), name="data_in")
    left = BatchNormalization(name="dynprop_norm")(dyn_inputs)

    seq_inputs = Input(shape=(seq_length,), dtype="int32", name="code_in")
    right = Embedding(
      output_dim=embedding_vector_length,
      input_dim=vocab_size,
      input_length=seq_length,
    )(seq_inputs)
    right = LSTM(16, consume_less="mem")(right)
    right = BatchNormalization(input_shape=(32,), name="lstm_norm")(right)

    aux_out = Dense(2, activation="sigmoid", name="aux_out")(right)

    x = merge([left, right], mode="concat")
    x = Dense(18, activation="relu")(x)
    out = Dense(2, activation="sigmoid", name="out")(x)

    model = Model(input=[dyn_inputs, seq_inputs], output=[out, aux_out])
    model.compile(
      optimizer="adam",
      loss={
        "out": "categorical_crossentropy",
        "aux_out": "categorical_crossentropy",
      },
      loss_weights={"out": 1.0, "aux_out": 0.2},
      metrics=["accuracy"],
    )
    return model

  def train_fn(model, train, *args, **kwargs):
    """ train a model """
    model.fit(
      {"data_in": train["x_2"], "code_in": train["x_seq"]},
      {"out": train["y_2"], "aux_out": train["y_2"]},
      nb_epoch=50,
      batch_size=batch_size,
      verbose=1,
      shuffle=True,
    )

  def test_fn(model, test, seed, *args, **kwargs):
    """ make predictions for test data """
    predictions = np.array(
      model.predict(
        {"data_in": test["x_2"], "code_in": test["x_seq"]},
        batch_size=batch_size,
        verbose=0,
      )
    )
    clipped = [np.argmax(x) for x in predictions[0]]
    return clipped

  def save_fn(outpath, model):
    """ save a trained model """
    model.save(outpath)

  def load_fn(inpath):
    """ load a trained model """
    return load_model(inpath)

  return {
    "name": "harry",
    "create_model": create_model,
    "train_fn": train_fn,
    "test_fn": test_fn,
    "save_fn": save_fn,
    "load_fn": load_fn,
  }


def karl():
  BATCH_SIZE = 64
  EMBEDDING_VECTOR_LEN = 64

  def create_model(*args, data_desc=None, **kwargs):
    """ instantiate a model """
    atomizer = data_desc["atomizer"]
    seq_length = data_desc["seq_length"]
    vocab_size = atomizer.vocab_size + 1

    data_in = Input(shape=(2,), name="data_in")

    code_in = Input(shape=(seq_length,), dtype="int32", name="code_in")
    right = Embedding(
      output_dim=EMBEDDING_VECTOR_LEN,
      input_dim=vocab_size,
      input_length=seq_length,
    )(code_in)
    right = LSTM(16, consume_less="mem")(right)

    aux_out = Dense(2, activation="sigmoid", name="aux_out")(right)

    x = merge([data_in, right], mode="concat")
    x = BatchNormalization(name="norm")(x)
    x = Dense(18, activation="relu")(x)
    out = Dense(2, activation="sigmoid", name="out")(x)

    model = Model(input=[data_in, code_in], output=[out, aux_out])
    model.compile(
      optimizer="adam",
      loss={
        "out": "categorical_crossentropy",
        "aux_out": "categorical_crossentropy",
      },
      loss_weights={"out": 1.0, "aux_out": 0.2},
      metrics=["accuracy"],
    )
    return model

  def train_fn(model, train, *args, **kwargs):
    """ train a model """
    model.fit(
      {"data_in": train["x_2"], "code_in": train["x_seq"]},
      {"out": train["y_2"], "aux_out": train["y_2"]},
      nb_epoch=50,
      batch_size=BATCH_SIZE,
      verbose=1,
      shuffle=True,
    )

  def test_fn(model, test, seed, *args, **kwargs):
    """ make predictions for test data """
    predictions = np.array(
      model.predict(
        {"data_in": test["x_2"], "code_in": test["x_seq"]},
        batch_size=BATCH_SIZE,
        verbose=0,
      )
    )
    clipped = [np.argmax(x) for x in predictions[0]]
    return clipped

  def save_fn(outpath, model):
    """ save a trained model """
    model.save(outpath)

  def load_fn(inpath):
    """ load a trained model """
    return load_model(inpath)

  return {
    "name": "karl",
    "create_model": create_model,
    "train_fn": train_fn,
    "test_fn": test_fn,
    "save_fn": save_fn,
    "load_fn": load_fn,
  }


def zero_r():
  def train_fn(model, train, *args, platform=None, **kwargs):
    """ train a model """
    if platform == "amd":
      model["zero_r"] = 0
    elif platform == "nvidia":
      model["zero_r"] = 1

  def test_fn(model, test, seed, *args, **kwargs):
    """ make predictions for test data """
    return [model["zero_r"]] * len(test["y"])

  def save_fn(outpath, model):
    """ save a trained model """
    with open(outpath, "wb") as outfile:
      pickle.dump(model, outfile)

  def load_fn(inpath):
    """ load a trained model """
    with open(inpath, "rb") as infile:
      model = pickle.load(infile)
    return model

  return {
    "name": "zero_r",
    "train_fn": train_fn,
    "test_fn": test_fn,
    "save_fn": save_fn,
    "load_fn": load_fn,
  }


def sally():
  """ multilayer """
  batch_size = 64

  def create_model(*args, data_desc=None, **kwargs):
    """ instantiate a model """
    atomizer = data_desc["atomizer"]
    seq_length = data_desc["seq_length"]

    embedding_vector_length = 64
    vocab_size = atomizer.vocab_size + 1

    dyn_inputs = Input(shape=(2,), name="data_in")

    seq_inputs = Input(shape=(seq_length,), dtype="int32", name="code_in")
    right = Embedding(
      output_dim=embedding_vector_length,
      input_dim=vocab_size,
      input_length=seq_length,
    )(seq_inputs)
    right = LSTM(16, consume_less="mem", return_sequences=True)(right)
    right = LSTM(16, consume_less="mem")(right)

    aux_out = Dense(2, activation="sigmoid", name="aux_out")(right)

    x = merge([dyn_inputs, right], mode="concat")
    x = BatchNormalization(name="norm")(x)
    x = Dense(18, activation="relu")(x)
    x = Dense(18, activation="relu")(x)
    out = Dense(2, activation="sigmoid", name="out")(x)

    model = Model(input=[dyn_inputs, seq_inputs], output=[out, aux_out])
    model.compile(
      optimizer="adam",
      loss={
        "out": "categorical_crossentropy",
        "aux_out": "categorical_crossentropy",
      },
      loss_weights={"out": 1.0, "aux_out": 0.2},
      metrics=["accuracy"],
    )
    return model

  def train_fn(model, train, *args, **kwargs):
    """ train a model """
    model.fit(
      {"data_in": train["x_2"], "code_in": train["x_seq"]},
      {"out": train["y_2"], "aux_out": train["y_2"]},
      nb_epoch=50,
      batch_size=batch_size,
      verbose=1,
      shuffle=True,
    )

  def test_fn(model, test, seed, *args, **kwargs):
    """ make predictions for test data """
    predictions = np.array(
      model.predict(
        {"data_in": test["x_2"], "code_in": test["x_seq"]},
        batch_size=batch_size,
        verbose=0,
      )
    )
    clipped = [np.argmax(x) for x in predictions[0]]
    return clipped

  def save_fn(outpath, model):
    """ save a trained model """
    model.save(outpath)

  def load_fn(inpath):
    """ load a trained model """
    return load_model(inpath)

  return {
    "name": "sally",
    "create_model": create_model,
    "train_fn": train_fn,
    "test_fn": test_fn,
    "save_fn": save_fn,
    "load_fn": load_fn,
  }


def donald():
  """ single layer output DNN """
  batch_size = 64

  def create_model(*args, data_desc=None, **kwargs):
    """ instantiate a model """
    atomizer = data_desc["atomizer"]
    seq_length = data_desc["seq_length"]

    embedding_vector_length = 64
    vocab_size = atomizer.vocab_size + 1

    dyn_inputs = Input(shape=(2,), name="data_in")

    seq_inputs = Input(shape=(seq_length,), dtype="int32", name="code_in")
    right = Embedding(
      output_dim=embedding_vector_length,
      input_dim=vocab_size,
      input_length=seq_length,
    )(seq_inputs)
    right = LSTM(16, consume_less="mem", return_sequences=True)(right)
    right = LSTM(16, consume_less="mem")(right)

    aux_out = Dense(2, activation="sigmoid", name="aux_out")(right)

    x = merge([dyn_inputs, right], mode="concat")
    x = BatchNormalization(name="norm")(x)
    x = Dense(18, activation="relu")(x)
    out = Dense(2, activation="sigmoid", name="out")(x)

    model = Model(input=[dyn_inputs, seq_inputs], output=[out, aux_out])
    model.compile(
      optimizer="adam",
      loss={
        "out": "categorical_crossentropy",
        "aux_out": "categorical_crossentropy",
      },
      loss_weights={"out": 1.0, "aux_out": 0.2},
      metrics=["accuracy"],
    )
    return model

  def train_fn(model, train, *args, **kwargs):
    """ train a model """
    model.fit(
      {"data_in": train["x_2"], "code_in": train["x_seq"]},
      {"out": train["y_2"], "aux_out": train["y_2"]},
      nb_epoch=50,
      batch_size=batch_size,
      verbose=1,
      shuffle=True,
    )

  def test_fn(model, test, seed, *args, **kwargs):
    """ make predictions for test data """
    predictions = np.array(
      model.predict(
        {"data_in": test["x_2"], "code_in": test["x_seq"]},
        batch_size=batch_size,
        verbose=0,
      )
    )
    clipped = [np.argmax(x) for x in predictions[0]]
    return clipped

  def save_fn(outpath, model):
    """ save a trained model """
    model.save(outpath)

  def load_fn(inpath):
    """ load a trained model """
    return load_model(inpath)

  return {
    "name": "donald",
    "create_model": create_model,
    "train_fn": train_fn,
    "test_fn": test_fn,
    "save_fn": save_fn,
    "load_fn": load_fn,
  }


def rupert():
  """ only dynprops """
  batch_size = 64

  def create_model(*args, data_desc=None, **kwargs):
    """ instantiate a model """
    atomizer = data_desc["atomizer"]
    seq_length = data_desc["seq_length"]

    dyn_inputs = Input(shape=(2,), name="data_in")
    x = BatchNormalization(name="norm")(dyn_inputs)
    x = Dense(2, activation="relu")(x)
    out = Dense(2, activation="sigmoid", name="out")(x)

    model = Model(input=dyn_inputs, output=out)
    model.compile(
      optimizer="adam",
      loss={"out": "categorical_crossentropy"},
      metrics=["accuracy"],
    )
    return model

  def train_fn(model, train, *args, **kwargs):
    """ train a model """
    model.fit(
      {"data_in": train["x_2"]},
      {"out": train["y_2"]},
      nb_epoch=50,
      batch_size=batch_size,
      verbose=1,
      shuffle=True,
    )

  def test_fn(model, test, seed, *args, **kwargs):
    """ make predictions for test data """
    predictions = np.array(
      model.predict(
        {"data_in": test["x_2"], "code_in": test["x_seq"]},
        batch_size=batch_size,
        verbose=0,
      )
    )
    clipped = [np.argmax(x) for x in predictions]
    return clipped

  def save_fn(outpath, model):
    """ save a trained model """
    model.save(outpath)

  def load_fn(inpath):
    """ load a trained model """
    return load_model(inpath)

  return {
    "name": "rupert",
    "create_model": create_model,
    "train_fn": train_fn,
    "test_fn": test_fn,
    "save_fn": save_fn,
    "load_fn": load_fn,
  }


def fred():
  BATCH_SIZE = 64
  EMBEDDING_VECTOR_LEN = 64

  def create_model(*args, data_desc=None, **kwargs):
    """ instantiate a model """
    atomizer = data_desc["atomizer"]
    seq_length = data_desc["seq_length"]
    vocab_size = atomizer.vocab_size + 1

    data_in = Input(shape=(2,), name="data_in")

    code_in = Input(shape=(seq_length,), dtype="int32", name="code_in")
    right = Embedding(
      output_dim=EMBEDDING_VECTOR_LEN,
      input_dim=vocab_size,
      input_length=seq_length,
    )(code_in)
    right = LSTM(16, consume_less="mem", return_sequences=True)(right)
    right = LSTM(16, consume_less="mem")(right)

    aux_out = Dense(2, activation="sigmoid", name="aux_out")(right)

    x = merge([data_in, right], mode="concat")
    x = BatchNormalization(name="norm")(x)
    x = Dense(18, activation="relu")(x)
    out = Dense(2, activation="sigmoid", name="out")(x)

    model = Model(input=[data_in, code_in], output=[out, aux_out])
    model.compile(
      optimizer="adam",
      loss={
        "out": "categorical_crossentropy",
        "aux_out": "categorical_crossentropy",
      },
      loss_weights={"out": 1.0, "aux_out": 0.2},
      metrics=["accuracy"],
    )
    return model

  def train_fn(model, train, *args, **kwargs):
    """ train a model """
    model.fit(
      {"data_in": train["x_2"], "code_in": train["x_seq"]},
      {"out": train["y_2"], "aux_out": train["y_2"]},
      nb_epoch=50,
      batch_size=BATCH_SIZE,
      verbose=1,
      shuffle=True,
    )

  def test_fn(model, test, seed, *args, **kwargs):
    """ make predictions for test data """
    predictions = np.array(
      model.predict(
        {"data_in": test["x_2"], "code_in": test["x_seq"]},
        batch_size=BATCH_SIZE,
        verbose=0,
      )
    )
    clipped = [np.argmax(x) for x in predictions[0]]
    return clipped

  def save_fn(outpath, model):
    """ save a trained model """
    model.save(outpath)

  def load_fn(inpath):
    """ load a trained model """
    return load_model(inpath)

  return {
    "name": "fred",
    "create_model": create_model,
    "train_fn": train_fn,
    "test_fn": test_fn,
    "save_fn": save_fn,
    "load_fn": load_fn,
  }


def barry():
  BATCH_SIZE = 64
  EMBEDDING_VECTOR_LEN = 64

  def create_model(*args, data_desc=None, **kwargs):
    """ instantiate a model """
    atomizer = data_desc["atomizer"]
    seq_length = data_desc["seq_length"]
    vocab_size = atomizer.vocab_size + 1

    data_in = Input(shape=(2,), name="data_in")

    code_in = Input(shape=(seq_length,), dtype="int32", name="code_in")
    right = Embedding(
      output_dim=EMBEDDING_VECTOR_LEN,
      input_dim=vocab_size,
      input_length=seq_length,
    )(code_in)
    right = LSTM(16, consume_less="mem")(right)

    aux_out = Dense(2, activation="sigmoid", name="aux_out")(right)

    x = merge([data_in, right], mode="concat")
    x = BatchNormalization(name="norm")(x)
    x = Dense(18, activation="relu")(x)
    x = Dense(18, activation="relu")(x)
    out = Dense(2, activation="sigmoid", name="out")(x)

    model = Model(input=[data_in, code_in], output=[out, aux_out])
    model.compile(
      optimizer="adam",
      loss={
        "out": "categorical_crossentropy",
        "aux_out": "categorical_crossentropy",
      },
      loss_weights={"out": 1.0, "aux_out": 0.2},
      metrics=["accuracy"],
    )
    return model

  def train_fn(model, train, *args, **kwargs):
    """ train a model """
    model.fit(
      {"data_in": train["x_2"], "code_in": train["x_seq"]},
      {"out": train["y_2"], "aux_out": train["y_2"]},
      nb_epoch=50,
      batch_size=BATCH_SIZE,
      verbose=1,
      shuffle=True,
    )

  def test_fn(model, test, seed, *args, **kwargs):
    """ make predictions for test data """
    predictions = np.array(
      model.predict(
        {"data_in": test["x_2"], "code_in": test["x_seq"]},
        batch_size=BATCH_SIZE,
        verbose=0,
      )
    )
    clipped = [np.argmax(x) for x in predictions[0]]
    return clipped

  def save_fn(outpath, model):
    """ save a trained model """
    model.save(outpath)

  def load_fn(inpath):
    """ load a trained model """
    return load_model(inpath)

  return {
    "name": "barry",
    "create_model": create_model,
    "train_fn": train_fn,
    "test_fn": test_fn,
    "save_fn": save_fn,
    "load_fn": load_fn,
  }


def turner():
  BATCH_SIZE = 64
  EMBEDDING_VECTOR_LEN = 32

  def create_model(*args, data_desc=None, **kwargs):
    """ instantiate a model """
    atomizer = data_desc["atomizer"]
    seq_length = data_desc["seq_length"]
    vocab_size = atomizer.vocab_size + 1

    data_in = Input(shape=(2,), name="data_in")

    code_in = Input(shape=(seq_length,), dtype="int32", name="code_in")
    right = Embedding(
      output_dim=EMBEDDING_VECTOR_LEN,
      input_dim=vocab_size,
      input_length=seq_length,
    )(code_in)
    right = LSTM(16, consume_less="mem")(right)

    aux_out = Dense(2, activation="sigmoid", name="aux_out")(right)

    x = merge([data_in, right], mode="concat")
    x = BatchNormalization(name="norm")(x)
    x = Dense(18, activation="relu")(x)
    out = Dense(2, activation="sigmoid", name="out")(x)

    model = Model(input=[data_in, code_in], output=[out, aux_out])
    model.compile(
      optimizer="adam",
      loss={
        "out": "categorical_crossentropy",
        "aux_out": "categorical_crossentropy",
      },
      loss_weights={"out": 1.0, "aux_out": 0.2},
      metrics=["accuracy"],
    )
    return model

  def train_fn(model, train, *args, **kwargs):
    """ train a model """
    model.fit(
      {"data_in": train["x_2"], "code_in": train["x_seq"]},
      {"out": train["y_2"], "aux_out": train["y_2"]},
      nb_epoch=50,
      batch_size=BATCH_SIZE,
      verbose=1,
      shuffle=True,
    )

  def test_fn(model, test, seed, *args, **kwargs):
    """ make predictions for test data """
    predictions = np.array(
      model.predict(
        {"data_in": test["x_2"], "code_in": test["x_seq"]},
        batch_size=BATCH_SIZE,
        verbose=0,
      )
    )
    clipped = [np.argmax(x) for x in predictions[0]]
    return clipped

  def save_fn(outpath, model):
    """ save a trained model """
    model.save(outpath)

  def load_fn(inpath):
    """ load a trained model """
    return load_model(inpath)

  return {
    "name": "turner",
    "create_model": create_model,
    "train_fn": train_fn,
    "test_fn": test_fn,
    "save_fn": save_fn,
    "load_fn": load_fn,
  }


def fife():
  BATCH_SIZE = 64
  EMBEDDING_VECTOR_LEN = 64

  def create_model(*args, data_desc=None, **kwargs):
    """ instantiate a model """
    atomizer = data_desc["atomizer"]
    seq_length = data_desc["seq_length"]
    vocab_size = atomizer.vocab_size + 1

    data_in = Input(shape=(2,), name="data_in")

    code_in = Input(shape=(seq_length,), dtype="int32", name="code_in")
    right = Embedding(
      output_dim=EMBEDDING_VECTOR_LEN,
      input_dim=vocab_size,
      input_length=seq_length,
    )(code_in)
    right = LSTM(16, consume_less="mem")(right)

    aux_out = Dense(2, activation="sigmoid", name="aux_out")(right)

    x = merge([data_in, right], mode="concat")
    x = Dense(18, activation="relu")(x)
    out = Dense(2, activation="sigmoid", name="out")(x)

    model = Model(input=[data_in, code_in], output=[out, aux_out])
    model.compile(
      optimizer="adam",
      loss={
        "out": "categorical_crossentropy",
        "aux_out": "categorical_crossentropy",
      },
      loss_weights={"out": 1.0, "aux_out": 0.2},
      metrics=["accuracy"],
    )
    return model

  def train_fn(model, train, *args, **kwargs):
    """ train a model """
    model.fit(
      {"data_in": train["x_2"], "code_in": train["x_seq"]},
      {"out": train["y_2"], "aux_out": train["y_2"]},
      nb_epoch=50,
      batch_size=BATCH_SIZE,
      verbose=1,
      shuffle=True,
    )

  def test_fn(model, test, seed, *args, **kwargs):
    """ make predictions for test data """
    predictions = np.array(
      model.predict(
        {"data_in": test["x_2"], "code_in": test["x_seq"]},
        batch_size=BATCH_SIZE,
        verbose=0,
      )
    )
    clipped = [np.argmax(x) for x in predictions[0]]
    return clipped

  def save_fn(outpath, model):
    """ save a trained model """
    model.save(outpath)

  def load_fn(inpath):
    """ load a trained model """
    return load_model(inpath)

  return {
    "name": "fife",
    "create_model": create_model,
    "train_fn": train_fn,
    "test_fn": test_fn,
    "save_fn": save_fn,
    "load_fn": load_fn,
  }


def bruno():
  BATCH_SIZE = 64
  EMBEDDING_VECTOR_LEN = 64

  def create_model(*args, data_desc=None, **kwargs):
    """ instantiate a model """
    atomizer = data_desc["atomizer"]
    seq_length = data_desc["seq_length"]
    vocab_size = atomizer.vocab_size + 1

    data_in = Input(shape=(2,), name="data_in")

    code_in = Input(shape=(seq_length,), dtype="int32", name="code_in")
    right = Embedding(
      output_dim=EMBEDDING_VECTOR_LEN,
      input_dim=vocab_size,
      input_length=seq_length,
    )(code_in)
    right = LSTM(64, consume_less="mem", return_sequences=True)(right)
    right = LSTM(64, consume_less="mem")(right)

    aux_out = Dense(2, activation="sigmoid", name="aux_out")(right)

    x = merge([data_in, right], mode="concat")
    x = BatchNormalization(name="norm")(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(2, activation="sigmoid", name="out")(x)

    model = Model(input=[data_in, code_in], output=[out, aux_out])
    model.compile(
      optimizer="adam",
      loss={
        "out": "categorical_crossentropy",
        "aux_out": "categorical_crossentropy",
      },
      loss_weights={"out": 1.0, "aux_out": 0.2},
      metrics=["accuracy"],
    )
    return model

  def train_fn(model, train, *args, **kwargs):
    """ train a model """
    model.fit(
      {"data_in": train["x_2"], "code_in": train["x_seq"]},
      {"out": train["y_2"], "aux_out": train["y_2"]},
      nb_epoch=50,
      batch_size=BATCH_SIZE,
      verbose=1,
      shuffle=True,
    )

  def test_fn(model, test, seed, *args, **kwargs):
    """ make predictions for test data """
    predictions = np.array(
      model.predict(
        {"data_in": test["x_2"], "code_in": test["x_seq"]},
        batch_size=BATCH_SIZE,
        verbose=0,
      )
    )
    clipped = [np.argmax(x) for x in predictions[0]]
    return clipped

  def save_fn(outpath, model):
    """ save a trained model """
    model.save(outpath)

  def load_fn(inpath):
    """ load a trained model """
    return load_model(inpath)

  return {
    "name": "bruno",
    "create_model": create_model,
    "train_fn": train_fn,
    "test_fn": test_fn,
    "save_fn": save_fn,
    "load_fn": load_fn,
  }


def brandon():
  BATCH_SIZE = 64
  EMBEDDING_VECTOR_LEN = 64

  def create_model(*args, data_desc=None, **kwargs):
    """ instantiate a model """
    atomizer = data_desc["atomizer"]
    seq_length = data_desc["seq_length"]
    vocab_size = atomizer.vocab_size + 1

    data_in = Input(shape=(2,), name="data_in")

    code_in = Input(shape=(seq_length,), dtype="int32", name="code_in")
    right = Embedding(
      output_dim=EMBEDDING_VECTOR_LEN,
      input_dim=vocab_size,
      input_length=seq_length,
    )(code_in)
    right = LSTM(64, consume_less="mem", return_sequences=True)(right)
    right = Dropout(0.1)(right)
    right = LSTM(64, consume_less="mem")(right)
    right = Dropout(0.1)(right)

    aux_out = Dense(2, activation="sigmoid", name="aux_out")(right)

    x = merge([data_in, right], mode="concat")
    x = BatchNormalization(name="norm")(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(2, activation="sigmoid", name="out")(x)

    model = Model(input=[data_in, code_in], output=[out, aux_out])
    model.compile(
      optimizer="adam",
      loss={
        "out": "categorical_crossentropy",
        "aux_out": "categorical_crossentropy",
      },
      loss_weights={"out": 1.0, "aux_out": 0.2},
      metrics=["accuracy"],
    )
    return model

  def train_fn(model, train, *args, **kwargs):
    """ train a model """
    model.fit(
      {"data_in": train["x_2"], "code_in": train["x_seq"]},
      {"out": train["y_2"], "aux_out": train["y_2"]},
      nb_epoch=50,
      batch_size=BATCH_SIZE,
      verbose=1,
      shuffle=True,
    )

  def test_fn(model, test, seed, *args, **kwargs):
    """ make predictions for test data """
    predictions = np.array(
      model.predict(
        {"data_in": test["x_2"], "code_in": test["x_seq"]},
        batch_size=BATCH_SIZE,
        verbose=0,
      )
    )
    clipped = [np.argmax(x) for x in predictions[0]]
    return clipped

  def save_fn(outpath, model):
    """ save a trained model """
    model.save(outpath)

  def load_fn(inpath):
    """ load a trained model """
    return load_model(inpath)

  return {
    "name": "brandon",
    "create_model": create_model,
    "train_fn": train_fn,
    "test_fn": test_fn,
    "save_fn": save_fn,
    "load_fn": load_fn,
  }


def janet():
  BATCH_SIZE = 64
  EMBEDDING_VECTOR_LEN = 64

  def create_model(*args, data_desc=None, **kwargs):
    """ instantiate a model """
    atomizer = data_desc["atomizer"]
    seq_length = data_desc["seq_length"]
    vocab_size = atomizer.vocab_size + 1

    data_in = Input(shape=(2,), name="data_in")

    code_in = Input(shape=(seq_length,), dtype="int32", name="code_in")
    right = Embedding(
      output_dim=EMBEDDING_VECTOR_LEN,
      input_dim=vocab_size,
      input_length=seq_length,
    )(code_in)
    right = LSTM(128, consume_less="mem", return_sequences=True)(right)
    right = LSTM(128, consume_less="mem")(right)

    aux_out = Dense(2, activation="sigmoid", name="aux_out")(right)

    x = merge([data_in, right], mode="concat")
    x = BatchNormalization(name="norm")(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(2, activation="sigmoid", name="out")(x)

    model = Model(input=[data_in, code_in], output=[out, aux_out])
    model.compile(
      optimizer="adam",
      loss={
        "out": "categorical_crossentropy",
        "aux_out": "categorical_crossentropy",
      },
      loss_weights={"out": 1.0, "aux_out": 0.2},
      metrics=["accuracy"],
    )
    return model

  def train_fn(model, train, *args, **kwargs):
    """ train a model """
    model.fit(
      {"data_in": train["x_2"], "code_in": train["x_seq"]},
      {"out": train["y_2"], "aux_out": train["y_2"]},
      nb_epoch=50,
      batch_size=BATCH_SIZE,
      verbose=1,
      shuffle=True,
    )

  def test_fn(model, test, seed, *args, **kwargs):
    """ make predictions for test data """
    predictions = np.array(
      model.predict(
        {"data_in": test["x_2"], "code_in": test["x_seq"]},
        batch_size=BATCH_SIZE,
        verbose=0,
      )
    )
    clipped = [np.argmax(x) for x in predictions[0]]
    return clipped

  def save_fn(outpath, model):
    """ save a trained model """
    model.save(outpath)

  def load_fn(inpath):
    """ load a trained model """
    return load_model(inpath)

  return {
    "name": "janet",
    "create_model": create_model,
    "train_fn": train_fn,
    "test_fn": test_fn,
    "save_fn": save_fn,
    "load_fn": load_fn,
  }
