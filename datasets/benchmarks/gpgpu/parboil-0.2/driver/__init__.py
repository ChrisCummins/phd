# (c) 2007 The Board of Trustees of the University of Illinois.

import sys
import os
from itertools import imap

import globals
import actions
import options
import parboilfile
import process
import benchmark

def run():
    # Print a banner message
    print "Parboil parallel benchmark suite, version 0.2"
    print
    
    # Global variable setup
    if not globals.root:
      globals.root = os.getcwd()

    python_path = (os.path.join(globals.root,'common','python') +
                   ":" +
                   os.environ.get('PYTHONPATH',""))

    bmks = parboilfile.Directory(os.path.join(globals.root,'benchmarks'), 
                     [], benchmark.benchmark_scanner())


    globals.benchdir = bmks

    globals.datadir =  parboilfile.Directory(
                         os.path.join(globals.root, 'datasets'), [], 
                         benchmark.dataset_repo_scanner())

    globals.benchmarks = benchmark.find_benchmarks()

    globals.program_env = {'PARBOIL_ROOT':globals.root,
                           'PYTHONPATH':python_path,
                           }

    # Parse options
    act = options.parse_options(sys.argv)

    # Perform the specified action
    if act:
        return act()

