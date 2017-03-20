#!/usr/bin/env python3
import fire
import pickle
import progressbar

from labm8 import fs
from pycparser.plyparser import ParseError
from pycparserext.ext_c_parser import OpenCLCParser
from random import shuffle


class Main(object):
    def parse(self, indir, outdir):
        """
        Parse OpenCL codes from indir, write binary parsed ASDs to outdir
        """
        indir = fs.path(indir)
        outdir = fs.path(outdir)
        fs.mkdir(outdir)

        # already done
        done = set(fs.basename(x) for x in fs.ls(outdir))
        # still to do
        filenames = list(set(fs.basename(x) for x in fs.ls(indir)) - done)
        # random order
        shuffle(filenames)

        print("{} input files".format(len(filenames)))

        bar = progressbar.ProgressBar()
        for filename in bar(filenames):
            inpath = fs.path(indir, filename)
            outpath = fs.path(outdir, filename)

            if not fs.exists(outpath):
                parser = OpenCLCParser()

                with open(inpath) as infile:
                    src = infile.read()

                try:
                    ast = parser.parse(src)
                    with open(outpath, "wb") as outfile:
                        pickle.dump(ast, outfile)
                except ParseError:
                    pass


if __name__ == "__main__":
    fire.Fire(Main)
