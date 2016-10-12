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

# End of sequence marker. This must be a character which does not exist in
# the training data.
EOF = '£'

def lr_encode(input, idx):
    input = input.strip()
    L = input[:idx]
    R = input[idx::][::-1]

    cmax = max(len(L), len(R))

    L = ' ' * (cmax - len(L)) + L
    R = ' ' * (cmax - len(R)) + R

    S = ''
    for i in range(cmax):
        S += L[i] + R[i]
    return S + EOF


def lr_decode(input):
    input = input.rstrip()
    assert(input[-1] == EOF)
    input = input[:-1]

    L = input[::2]
    R = input[1::2][::-1]

    print("L:", L)
    print("R:", R)

    assert(len(L) == len(R))

    return L.lstrip() + R
