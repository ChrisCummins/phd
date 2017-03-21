
# coding: utf-8

# # Adversarial Network Tuning
# 
# Definitions:
# * Network parameters: $\Theta$
# * CLgen model: $G(\Theta)$
# * Discriminator model: $D(G(\Theta))$
# 
# From candidate params $\Theta = \{\Theta_1, \Theta_2, \ldots, \Theta_n\}$,
# find the params $\Theta_{i}$ which minimize accuracy of the discriminator.
# 
# Discriminator functions:
# * Human-or-robot? Distinguish between programs from GitHub and synthesized codes.
# * Is it *useful*? Determine closest distance to benchmark features (would have to be a different set of benchmarks).
# 
# 
# 1. $\epsilon = 0.05$
# 1. $\Theta = newParams()$
# 1. while $abs(D(G(\Theta)) - 0.5) > \epsilon$
# 1. `    ` $\Theta = newParams()$

# ## GitHub Corpus

# In[1]:

from clgen.corpus import Corpus

corpus = Corpus.from_json({
    "path": "~/data/github",
    "vocabulary": "greedy"
})
corpus


# In[2]:

import sqlite3

def corpus_iter(kernels_db):
    """ fetch preprocessed source codes """
    db = sqlite3.connect(kernels_db)
    c = db.cursor()
    c.execute("SELECT contents FROM PreprocessedFiles WHERE status=0")
    srcs = [row[0] for row in c.fetchall()]
    c.close()
    db.close()
    return srcs


# In[3]:

import pickle
from labm8 import fs

github_srcs = corpus_iter(fs.path(corpus.contentcache.path, "kernels.db"))

inpath = fs.path("data", "encoded-" + corpus.hash + ".pkl")
if fs.exists(inpath):
    with open(inpath, "rb") as infile:
        github_seqs = pickle.load(infile)
else:
    github_seqs = [corpus.atomizer.atomize(x) for x in github_srcs]
    with open(inpath, "wb") as outfile:
        pickle.dump(github_seqs, outfile)
        print("cached", inpath)


# In[4]:

import numpy as np
import pandas as pd

lens = np.array([len(x) for x in github_seqs])
_data = [{"Percentile": x, "Sequence Length": int(round(np.percentile(lens, x)))} for x in range(0, 101, 10)]
data = pd.DataFrame(_data, columns=["Percentile", "Sequence Length"])
data


# In[5]:

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole?
            return True
        elif shell == 'TerminalInteractiveShell':  # Terminal running IPython?
            return False
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if isnotebook():
    import matplotlib.pyplot as plt
    import seaborn as sns
    get_ipython().magic('matplotlib inline')

    plt.semilogy(data["Percentile"], data["Sequence Length"])
    plt.title("GitHub Corpus")
    plt.xlabel("Percentile")
    plt.ylabel("Sequence Length")


# ## Synthetic Corpus

# In[6]:

syn_corpus = Corpus.from_json({
    "path": "~/data/synthetic-2017-02-01",
    "vocabulary": "greedy"
})
syn_corpus


# In[7]:

syn_srcs = corpus_iter(fs.path(syn_corpus.contentcache.path, "kernels.db"))

inpath = fs.path("data", "encoded-" + syn_corpus.hash + ".pkl")
if fs.exists(inpath):
    with open(inpath, "rb") as infile:
        syn_seqs = pickle.load(infile)
else:
    syn_seqs = [corpus.atomizer.atomize(x) for x in syn_srcs]
    with open(inpath, "wb") as outfile:
        pickle.dump(syn_seqs, outfile)
        print("cached", inpath)


# ## Discriminator Model

# Encoded sequence length:

# In[8]:

import scipy.stats

seq_length = 1024
p1 = scipy.stats.percentileofscore(lens, seq_length)
p2 = 100 - p1
print("""A sequence length of {seq_length} is the {p1:.1f}% percentile of the GitHub corpus.
{p2:.1f}% of sequences will be truncated.""".format(**vars()))


# In[38]:

seed = 204

# inputs
vocab_size = corpus.vocab_size + 1  # pad value

# network param
embedding_vector_length = 64
lstm_size = 64

# training param
test_split = .2
validation_split = .2  # note this is the split of the training set
nb_epoch = 50
batch_size = 128


# Assemble dataset:

# In[105]:

import setGPU
from keras.preprocessing.sequence import pad_sequences
import random

# re-prodicuble results:
random.seed(seed)
np.random.seed(seed)

# pad sequences and label '1' for synthetic, '0' for github
pad_val = vocab_size - 1

# shuffle sequences before truncating to nmax
nmax = min(len(github_seqs), len(syn_seqs))
random.shuffle(github_seqs)
random.shuffle(syn_seqs)

X = list(pad_sequences(github_seqs[:nmax], maxlen=seq_length, value=pad_val))
y = [np.array([1, 0])] * len(github_seqs[:nmax])  # 1-hot encoding

X += list(pad_sequences(syn_seqs[:nmax], maxlen=seq_length, value=pad_val))
y += [np.array([0, 1])] * len(syn_seqs[:nmax])  # 1-hot encoding

dataset = list(zip(X, y))
n = len(dataset)

X = np.array(X)
y = np.array(y)

print("Dataset of {n} instances ({nmax} of each type).".format(**vars()))


# Split into train and test sets:

# In[112]:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

n_train = len(X_train)
n_train_gh = sum(1 if x[0] == 1 else 0 for x in y_train)
n_train_syn = sum(1 if x[1] == 1 else 0 for x in y_train)
train_syn_ratio = n_train_syn / n_train
assert(n_train_gh + n_train_syn == n_train)

n_val = int(len(X_train) * validation_split)

n_test = len(X_test)
n_test_gh = sum(1 if x[0] == 1 else 0 for x in y_test)
n_test_syn = sum(1 if x[1] == 1 else 0 for x in y_test)
test_syn_ratio = n_test_syn / n_test

print("""Dataset of {n} instances ({nmax} of each type).
{n_train} instances for training ({n_val} of those for validation).
{n_test} instances for testing. Ratio of synthetic: {test_syn_ratio:.1%}

Ratio of synthetic codes: {train_syn_ratio:.1%} train, {test_syn_ratio:.1%} test.\
""".format(**vars()))


# Discriminator architecture:

# In[113]:

from keras.layers import Input, Dropout, Embedding, LSTM, Dense
from keras.models import Model, load_model
from keras.utils.visualize_util import model_to_dot
from IPython.display import SVG

def create_model():
    """ instantiate model """
    data_in = Input(shape=(2,), name="data_in")

    code_in = Input(shape=(seq_length,), dtype="int32", name="code_in")
    x = Embedding(output_dim=embedding_vector_length, input_dim=vocab_size, input_length=seq_length)(code_in)
    x = LSTM(lstm_size, input_dim=vocab_size, input_length=seq_length, consume_less="mem", return_sequences=True)(x)
    x = LSTM(lstm_size, consume_less="mem")(x)
    out = Dense(2, activation="sigmoid", name="out")(x)

    model = Model(input=code_in, output=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    return model

_m = create_model()
_m.summary()
SVG(model_to_dot(_m, show_shapes=True).create(prog='dot', format='svg'))


# Create and train model:

# In[92]:

model_path = fs.path("data", "model.h5")
if fs.exists(model_path):
    model = load_model(model_path)
else:
    model = create_model()
    
    np.random.seed(seed)
    model.fit(X_train, y_train, nb_epoch=nb_epoch, validation_split=validation_split,
              batch_size=batch_size, verbose=1, shuffle=True)

    model.save(model_path)
    print("cached", model_path)


# Test model:

# In[57]:

predictions_path = fs.path("data", "predictions.pkl")
if fs.exists(predictions_path):
    with open(predictions_path, "rb") as infile:
        predictions = pickle.load(infile)
else:
    predictions = np.array(model.predict(X_test, batch_size=batch_size, verbose=0))
    with open(predictions_path, "wb") as outfile:
        pickle.dump(predictions, outfile)


# Evaluate results:

# In[81]:

true_syn = sum(1 for x in y_test if x[1] == 1)
true_gh = sum(1 for x in y_test if x[0] == 1)
assert(true_syn + true_gh == len(y_test))

clipped = [(0, 1) if y > x else (1, 0) for x, y in predictions]
correct = [1 if x[0] == y[0] and x[1] == y[1] else 0 for (x, y) in zip(clipped, y_test)]
n_correct = sum(correct)

score = n_correct / n_test
print("""Discriminator correctly classified {n_correct} of {n_test} instances ({score:.1%})""".format(**vars()))


# In[86]:

data = pd.DataFrame([
    {"source": "Truth", "Kernel source": "Synthetic", "Y": sum(x[1] for x in y_test)},
    {"source": "Truth", "Kernel source": "Real", "Y": sum(x[0] for x in y_test)},
    {"source": "Discriminator", "Kernel source": "Synthetic", "Y": sum(x[1] for x in clipped)},
    {"source": "Discriminator", "Kernel source": "Real", "Y": sum(x[0] for x in clipped)},
])
sns.barplot(data=data, x="Kernel source", y="Y", hue="source")
plt.ylabel("#. kernels")
plt.legend(title="")


# In[ ]:



