#
# cgo13 - Implementation of the autotuner from:
#
#     Grewe, D., Wang, Z., & O’Boyle, M. F. P. M. (2013). Portable
#     Mapping of Data Parallel Programs to OpenCL for Heterogeneous
#     Systems. In CGO. IEEE.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import smith

from re import sub

# End of sequence marker. This must be a character which does not exist in
# the training data.
EOF = '£'

PAD = '±'

def lr_encode(input, idx):
    input = input.strip()
    L = input[:idx]
    R = input[idx::][::-1]

    cmax = max(len(L), len(R))

    L = PAD * (cmax - len(L)) + L
    R = PAD * (cmax - len(R)) + R

    S = ''
    for i in range(cmax):
        S += L[i] + R[i]
    return S + EOF


def lr_decode(input):
    input = input.strip()
    assert(input[-1] == EOF)
    assert(input.count(EOF) == 1)
    input = input[:-1]

    L = input[::2]
    R = input[1::2][::-1]

    assert(len(L) == len(R) or len(L) - 1 == len(R))

    return sub('^{}+'.format(PAD), '', L) + sub('{}+$'.format(PAD), '', R)
