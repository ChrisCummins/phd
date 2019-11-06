from typing import Union, List

import numpy as np
import tensorflow as tf
from labm8 import app

FLAGS = app.FLAGS

def pos_emb(positions: Union[int, List[int], np.array], demb: int = 200):
    """Transformer-like sinusoidal positional embeddings.
        Args:
        position: int or array of ints   positions to embed,
        demb: int    size of embedding vector
    """
    inv_freq = 1 / (10000 ** (np.arange(0.0, demb, 2.0) / demb))

    sinusoid_inp = np.outer(positions, inv_freq)
    pos_emb = np.hstack((np.sin(sinusoid_inp), np.cos(sinusoid_inp)))
    return pos_emb