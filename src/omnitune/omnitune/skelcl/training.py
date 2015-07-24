import itertools
import random

import labm8 as lab
from labm8 import io

from . import hash_params


def random_wg_value(max_wg_size):
    wg_c = random.randrange(2, max_wg_size, 2)
    wg_r = random.randrange(2, max_wg_size, 2)

    if wg_c * wg_r > max_wg_size:
        wg_c = random.randrange(2, max_wg_size, 2)
        wg_r = random.randrange(2, max_wg_size, 2)
    return [wg_c, wg_r]
