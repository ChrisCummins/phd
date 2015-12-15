#!/usr/bin/env python2
"""
Parse the output of texcount, printing a table of <file> <wc>
pairs. Example usage:

    $ parse_texcount.py < texcout.txt

                             thesis     0
                           document     2
                              draft     0
                           preamble   694
                        preliminary     0
                           abstract   524
                  chap/introduction   713
                    chap/background  1021
                       chap/related  1390
                   chap/methodology  2385
                            alg/gol    16
                          tab/hosts     4
                        tab/devices     4
                        tab/kernels     4
                       tab/datasets     4
     tab/stencil-runtime-components     4
                       fig/speedups     0
                       fig/heatmaps     2
            fig/performance-wgsizes     2
                   fig/performances     2
                      chap/autotune  1684
                       tab/features     9
        alg/autotune-classification    35
                alg/autotune-hybrid    55
                    chap/evaluation   170
                   chap/conclusions     0
                              Total  8724

You can specify the kinds of file paths to match by setting the
TEXCOUNT_MATCH environment variable. Excluded files do not count
towards the total.

    $ TEXCOUNT_MATCH='^chap/' parse_texcount.py < texcout.txt

     chap/introduction   713
       chap/background  1021
          chap/related  1390
      chap/methodology  2385
         chap/autotune  1684
       chap/evaluation   170
      chap/conclusions     0
                 Total  7363
"""

from __future__ import print_function
from os import environ
from re import compile,search,sub
from sys import stdin,stdout

import labm8
from labm8 import fmt

_FILE_RE = compile("^File: (.+)")
_WORDS_IN_TEXT_RE = compile("^Words in text: (\d+)")


def parse_wc(input, output=stdout, match_pattern=None):
    match_re = compile(match_pattern) if match_pattern else None

    files = []

    # State.
    infile = False
    file = None

    # Iterate over lines in the input.
    for line in input.readlines():
        if infile:
            match = search(_WORDS_IN_TEXT_RE, line)
            if match:
                files.append((file, int(match.group(1))))
                infile = False
        else:
            match = search(_FILE_RE, line)
            if match:
                file = sub(".tex$", "", match.group(1))
                if match_re:
                    if search(match_re, file):
                        infile = True
                else:
                    infile = True

    total = sum([x[1] for x in files])
    files.append(("Total", total))

    # Print word counts.
    print(fmt.table(files, justify="left"))


def main():
    # Set the input and output files.
    input = stdin
    output = stdout

    parse_wc(input, output, match_pattern=environ.get('TEXCOUNT_MATCH'))


if __name__ == "__main__":
    main()
