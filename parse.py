#!/usr/bin/env python3
import fire
import pickle

from labm8 import fs
from pycparser.plyparser import ParseError
from pycparserext.ext_c_generator import OpenCLCGenerator
from pycparserext.ext_c_parser import OpenCLCParser
from random import shuffle


class Main(object):
    def parse(self, indir, outdir):
        """
        Parse OpenCL codes from indir, write binary parsed ASDs to outdir
        """
        indir = fs.path(indir)
        outdir = fs.path(outdir)

        files = fs.ls(indir, abspaths=True)
        n = len(files)
        print("{n} input files".format(**vars()))

        shuffle(files)

        fs.mkdir(outdir)

        for i, path in enumerate(files):
            if i % 10 == 0:
                print("{i}/{n}".format(**vars()))
            parser = OpenCLCParser()
            with open(path) as infile:
                src = infile.read()

            try:
                ast = parser.parse(src)
                outpath = fs.path(outdir, fs.basename(path))
                with open(outpath, "wb") as outfile:
                    pickle.dump(ast, outfile)
            except ParseError:
                print(i + 1, "error", fs.basename(path))


if __name__ == "__main__":
    fire.Fire(Main)
