#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Command line interface to clgen.
"""
import cProfile
import argparse
import os
import sys
import traceback

from argparse import RawTextHelpFormatter
from labm8 import jsonutil, fs, prof
from sys import exit
from typing import List

import clgen
from clgen import log
from clgen import dbutil


def print_version_and_exit():
    """
    Print the clgen version. This function does not return.
    """
    version = clgen.version()
    print(f"clgen {version} made with \033[1;31m♥\033[0;0m by "
          "Chris Cummins <chrisc.101@gmail.com>.")
    exit(0)


class ArgumentParser(argparse.ArgumentParser):
    """
    CLgen specialized argument parser.

    Differs from python argparse.ArgumentParser in the following areas:
      * Adds an optional `--verbose` flag and initializes the logging engine.
      * Adds a `--debug` flag for more verbose crashes.
      * Adds a `--profile` flag for internal profiling.
      * Adds an optional `--version` flag which prints version information and
        quits.
      * Defaults to using raw formatting for help strings, which disables line
        wrapping and whitespace squeezing.
      * Appends author information to description.
    """
    def __init__(self, *args, **kwargs):
        """
        See python argparse.ArgumentParser.__init__().
        """
        # append author information to description
        description = kwargs.get("description", "")

        if len(description) and description[-1] != "\n":
            description += "\n"
        description += """
Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
<http://chriscummins.cc/clgen>"""

        kwargs["description"] = description.lstrip()

        # unless specified otherwise, use raw text formatter. This means
        # that whitespace isn't squeed and lines aren't wrapped
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = RawTextHelpFormatter

        # call built in ArgumentParser constructor.
        super(ArgumentParser, self).__init__(*args, **kwargs)

        # Add defualt arguments
        self.add_argument("--version", action="store_true",
                          help="show version information and exit")
        self.add_argument("-v", "--verbose", action="store_true",
                          help="increase output verbosity")
        self.add_argument("--debug", action="store_true",
                          help="in case of error, print debugging information")
        self.add_argument("--profile", action="store_true",
                          help="enable internal API profiling")

    def parse_args(self, args=sys.argv[1:], namespace=None):
        """
        See python argparse.ArgumentParser.parse_args().
        """
        # --version option overrides the normal argument parsing process.
        if "--version" in args:
            print_version_and_exit()

        # parse args normally
        args_ns = super(ArgumentParser, self).parse_args(args, namespace)

        # set log level
        log.init(args_ns.verbose)

        # set debug option
        if args_ns.debug:
            os.environ["DEBUG"] = "1"

        # set profile option
        if args_ns.profile:
            prof.enable()

        return args_ns


def main(method, *args, **kwargs):
    """
    Runs the given method as the main entrypoint to a program.

    If an exception is thrown, print error message and exit.

    If environmental variable DEBUG=1, then exception is not caught.

    Args:
        method (function): Function to execute.
        *args (str): Arguments for method.
        **kwargs (dict): Keyword arguments for method.

    Returns:
        method(*args, **kwargs)
    """
    def _user_message(exception):
        log.fatal("""\
{err} ({type})

Please report bugs at <https://github.com/ChrisCummins/clgen/issues>\
""".format(err=e, type=type(e).__name__))

    def _user_message_with_stacktrace(exception):
        # get limited stack trace
        def _msg(i, x):
            n = i + 1

            filename = fs.basename(x[0])
            lineno = x[1]
            fnname = x[2]

            loc = "{filename}:{lineno}".format(**vars())
            return "      #{n}  {loc: <18} {fnname}()".format(**vars())

        _, _, tb = sys.exc_info()
        NUM_ROWS = 5  # number of rows in traceback

        trace = reversed(traceback.extract_tb(tb, limit=NUM_ROWS+1)[1:])
        message = "\n".join(_msg(*r) for r in enumerate(trace))

        log.fatal("""\
{err} ({type})

  stacktrace:
{stack_trace}

Please report bugs at <https://github.com/ChrisCummins/clgen/issues>\
""".format(err=e, type=type(e).__name__, stack_trace=message))

    # if DEBUG var set, don't catch exceptions
    if os.environ.get("DEBUG", None):
        # verbose stack traces. see: https://pymotw.com/2/cgitb/
        import cgitb
        cgitb.enable(format='text')

        return method(*args, **kwargs)

    try:
        def run():
            method(*args, **kwargs)

        if prof.is_enabled():
            return cProfile.runctx('run()', None, locals(), sort='tottime')
        else:
            return run()
    except clgen.UserError as err:
        log.fatal(err, "(" + type(err).__name__  + ")")
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stderr.flush()
        print("\nkeyboard interrupt, terminating", file=sys.stderr)
        sys.exit(1)
    except clgen.UserError as e:
        _user_message(e)
    except clgen.File404 as e:
        _user_message(e)
    except Exception as e:
        _user_message_with_stacktrace(e)


def atomize(args: List[str]=sys.argv[1:]):
    """
    Extract and print corpus vocabulary.
    """
    def _atomize(path, vocab, size=False):
        with open(clgen.must_exist(path)) as infile:
            data = infile.read()
        atoms = corpus.atomize(data, vocab=vocab)

        if size:
            log.info("size:", len(atoms))
        else:
            log.info('\n'.join(atoms))

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input', help='path to input text file')
    parser.add_argument('-t', '--type', type=str, default='char',
                        help='vocabulary type')
    parser.add_argument('-s', '--size', action="store_true",
                        help="print vocabulary size")
    args = parser.parse_args(args)

    opts = {
        "path": args.input,
        "vocab": args.type,
        "size": args.size
    }
    main(_atomize, **opts)


def train(args: List[str]=sys.argv[1:]):
    """
    Train a CLgen model.
    """
    def _train(model_path: str):
        model_json = jsonutil.read_file(model_path)
        model = clgen.Model.from_json(model_json)
        model.train()
        print("done.")

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", metavar="<model>",
                        help="path to model dist or specification file")
    args = parser.parse_args(args)

    main(_train, args.model)


def test(args: List[str]=sys.argv[1:]):
    """
    Run the CLgen self-test suite.
    """
    import clgen.test  # Must scope this import, as it breaks cache behaviour

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--coveragerc-path", action="store_true",
                        help="print path to coveragerc file")
    parser.add_argument("--coverage-path", action="store_true",
                        help="print path to coverage file")
    args = parser.parse_args(args)

    if args.coveragerc_path:
        print(clgen.test.coveragerc_path())
        sys.exit(0)

    if args.coverage_path:
        print(clgen.test.coverage_report_path())
        sys.exit(0)

    sys.exit(clgen.test.main())


def fetch(args: List[str]=sys.argv[1:]):
    """
    Import OpenCL files into kernel datbase.

    The kernel database is used as a staging ground for input files, which are
    then preprocessed and assembled into corpuses. This program acts as the front
    end, assembling files from the file system into a database for preprocessing.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input', help='path to SQL dataset')
    parser.add_argument('paths', type=str, nargs='+',
                        help='path to OpenCL files or directories')
    args = parser.parse_args(args)

    db_path = os.path.expanduser(args.input)

    main(clgen.fetch, db_path, args.paths)
    log.info("done.")


def fetch_github(args: List[str]=sys.argv[1:]):
    """
    Mines OpenCL kernels from Github. Requires the following environment
    variables to be set:

         GITHUB_USERNAME   github username
         GITHUB_PW         github password
         GITHUB_TOKEN      github api token

    For instructions to generate an API token, see:

      <https://help.github.com/articles/creating-an-access-token-for-command-line-use/>

    This process issues thousands of GitHub API requests per minute. Please
    exercise restrained in minimizing your use of this program -- we don't
    want to upset the nice folks at GH :-)
    """
    from os import environ
    from github import BadCredentialsException

    def _fetch(*args, **kwargs):
        try:
            fetch_github(*args, **kwargs)
        except BadCredentialsException as e:
            log.fatal("bad GitHub credentials")

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input', help='path to SQL input dataset')
    args = parser.parse_args(args)

    db_path = args.input

    try:
        github_username = environ['GITHUB_USERNAME']
        github_pw = environ['GITHUB_PW']
        github_token = environ['GITHUB_TOKEN']
    except KeyError as e:
        log.fatal('environment variable {} not set'.format(e))

    main(_fetch, db_path, github_username, github_pw, github_token)


def refresh_cache(args: List[str]=sys.argv[1:]):
    """
    Refresh the cached model, corpus, and sampler IDs.
    """
    parser = ArgumentParser(description=__doc__)
    args = parser.parse_args(args)

    cache = clgen.cachepath()

    log.warning("Not Implemented: refresh corpuses")

    if fs.isdir(cache, "model"):
        for cached_modeldir in fs.ls(fs.path(cache, "model"), abspaths=True):
            cached_model_id = fs.basename(cached_modeldir)
            cached_meta = jsonutil.read_file(fs.path(cached_modeldir, "META"))

            model = clgen.Model.from_json(cached_meta)

            if cached_model_id != model.hash:
                log.info(cached_model_id, '->', model.hash)

                if fs.isdir(model.cache.path):
                    log.error("cache conflict", file=sys.stderr)
                    sys.exit(1)

                fs.mv(cached_modeldir, model.cache.path)

    log.warning("Not Implemented: refresh samplers")


def preprocess(args: List[str]=sys.argv[1:]):
    """
    Process OpenCL files for machine learning.

    This is a three step process. First, the OpenCL kernels are compiled to
    bytecode, then the source files are preprocessed, before being rewritten.

    Preprocessing is computationally demanding and highly paralellised.
    Expect high resource contention during preprocessing.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', help='path to input')
    parser.add_argument('-f', '--file', action='store_true',
                        help='treat input as file')
    parser.add_argument('-i', '--inplace', action='store_true',
                        help='inplace file rewrite')
    parser.add_argument('-G', '--gpuverify', action='store_true',
                        help='run GPUVerify on kernels')
    parser.add_argument('--remove-bad-preprocessed', action='store_true',
                        help="""
delete the contents of all bad or ugly preprocessed files,
but keep the entries in the table""".strip())
    parser.add_argument("--remove-preprocessed", action="store_true",
                        help="remove all preprocessed files from database")
    args = parser.parse_args(args)

    preprocess_opts = {
        "use_gpuverify": args.gpuverify,
    }

    if args.file and args.inplace:
        main(clgen.preprocess_inplace, args.inputs, **preprocess_opts)
    else:
        for path in args.inputs:
            if args.file:
                main(clgen.preprocess_file, path,
                     inplace=False, **preprocess_opts)
            elif args.remove_bad_preprocessed:
                main(dbutil.remove_bad_preprocessed, path)
            elif args.remove_preprocessed:
                main(dbutil.remove_preprocessed, path)
                log.info("done.")
            else:
                if main(clgen.preprocess_db, path, **preprocess_opts):
                    log.info("done.")
                else:
                    log.info("nothing to be done.")


def merge(args: List[str]=sys.argv[1:]):
    """
    Merge kernel datasets.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="path to output dataset")
    parser.add_argument("inputs", nargs='*', help="path to input datasets")
    args = parser.parse_args(args)

    main(dbutil.merge, args.dataset, args.inputs)


def grid(args: List[str]=sys.argv[1:]):
    """
    Print model stats.
    """
    from prettytable import PrettyTable

    parser = ArgumentParser(description=__doc__)
    args = parser.parse_args(args)

    cache = clgen.cachepath()

    x = PrettyTable([
        "id",
        "corpus",
        "trained",
        "type",
        "nodes",
        "epochs",
        "lr",
        "dr",
        "gc",
    ])

    x.align['nodes'] = 'r'

    for model in clgen.models():
        meta = model.to_json()

        network = f'{meta["architecture"]["rnn_size"]} x {meta["architecture"]["num_layers"]}'

        if "stats" in meta:
            num_epochs = len(meta["stats"]["epoch_costs"])
        else:
            num_epochs = 0

        if num_epochs >= meta["train_opts"]["epochs"]:
            trained = "Y"
        elif fs.isfile(fs.path(model.cache.path, "LOCK")):
            trained = f"WIP ({num_epochs}/{meta['train_opts']['epochs']})"
        elif num_epochs > 0:
            trained = f"{num_epochs}/{meta['train_opts']['epochs']}"
        else:
            trained = ""

        x.add_row([
            model.shorthash,
            model.shorthash,
            trained,
            meta["architecture"]["model_type"],
            network,
            meta["train_opts"]["epochs"],
            "{:.0e}".format(meta["train_opts"]["learning_rate"]),
            meta["train_opts"]["lr_decay_rate"],
            meta["train_opts"]["grad_clip"],
        ])

    print(x.get_string(sortby="nodes"))


def features(args: List[str]=sys.argv[1:]):
    """
    Extract static OpenCL kernel features.

    This extracts the static compile-time features of the paper:

        Grewe, D., Wang, Z., & O'Boyle, M. F. P. M. (2013). Portable Mapping of
        Data Parallel Programs to OpenCL for Heterogeneous Systems. In CGO. IEEE.
    """
    from clgen import features

    def features_dir(csv_path):
        return fs.basename(fs.dirname(csv_path))

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', help='input path(s)')
    parser.add_argument('-d', '--dir-mode', action='store_true',
                        help='treat inputs as directories')
    parser.add_argument('-s', '--stats', action='store_true',
                        help='summarize a features files')
    parser.add_argument('-e', '--fatal-errors', action='store_true',
                        help='quit on compiler error')
    parser.add_argument('--shim', action='store_true',
                        help='include shim header')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='minimal error output')
    parser.add_argument('-H', '--no-header', action='store_true',
                        help='no features header')
    args = parser.parse_args(args)

    inputs = args.inputs
    dir_mode = args.dir_mode
    summarise = args.stats

    if summarise:
        stats = [features.summarize(f) for f in inputs]

        print('dataset', *list(stats[0].keys()), sep=',')
        for path, stat in zip(inputs, stats):
            print(features_dir(path), *list(stat.values()), sep=',')
        return

    if dir_mode:
        trees = [fs.ls(d, abspaths=True, recursive=True) for d in inputs]
        paths = [item for sublist in trees for item in sublist]
    else:
        paths = [fs.path(f) for f in inputs]

    main(features.files, paths, fatal_errors=args.fatal_errors,
         use_shim=args.shim, quiet=args.quiet, header=not args.no_header)


def explore(args: List[str]=sys.argv[1:]):
    """
    Exploratory analysis of preprocessed dataset.

    Provides an overview of the contents of an OpenCL kernel database.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input', help='path to SQL input dataset')
    args = parser.parse_args(args)

    db_path = args.input

    main(clgen.explore, db_path)


def dump(args: List[str]=sys.argv[1:]):
    """
    Dump kernel dataset to file(s).
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input', help='path to kernels database')
    parser.add_argument('output', help='path to output file or directory')
    parser.add_argument('-d', action='store_true', default=False,
                        help='output to directory (overrides -i, --eof, -r)')
    parser.add_argument('-i', action='store_true', default=False,
                        help='include file separators')
    parser.add_argument('--input-samples', action='store_true',
                        default=False,
                        help='use input contents, not preprocessed')
    parser.add_argument('--eof', action='store_true', default=False,
                        help='print end of file')
    parser.add_argument('-r', action='store_true', default=False,
                        help='use reverse order')
    parser.add_argument('-s', '--status', type=int, default=0,
                        help='status code to use')
    args = parser.parse_args(args)

    db_path = args.input
    out_path = args.output
    opts = {
        "dir": args.d,
        "eof": args.eof,
        "fileid": args.i,
        "input_samples": args.input_samples,
        "reverse": args.r,
        "status": args.status
    }

    main(dbutil.dump_db, db_path, out_path, **opts)


def create_db(args: List[str]=sys.argv[1:]):
    """
    Create an empty OpenCL kernel database.

    THIS IS A TEST.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input', help='path to SQL input dataset')
    parser.add_argument('-g', '--github', action='store_true',
                        help='generate dataset with GitHub metadata')
    args = parser.parse_args(args)

    main(dbutil.create_db, args.input, args.github)
    log.info(fs.abspath(args.input))


def clgen_main(args: List[str]=sys.argv[1:]):
    """
    Generate OpenCL programs using Deep Learning.

    This is a five-step process:
       1. Input files are collected from the model specification file.
       2. The input files are preprocessed into an OpenCL kernel database.
       3. A training corpus is generated from the input files.
       4. A model is instantiated and trained on the corpus.
       5. The trained model is sampled for new kernels.

    This program automates the execution of all five stages of the pipeline.
    The pipeline can be interrupted and resumed at any time. Results are cached
    across runs.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", metavar="<model>",
                        help="path to model dist or specification file")
    parser.add_argument("sampler", metavar="<sampler>",
                        help="path to sampler specification file")
    parser.add_argument("--ls-files", action="store_true",
                        help="print cached corpus, model, and sampler, files")
    parser.add_argument("--corpus-dir", action="store_true",
                        help="print path to corpus cache")
    parser.add_argument("--model-dir", action="store_true",
                        help="print path to model cache")
    parser.add_argument("--sampler-dir", action="store_true",
                        help="print path to sampler cache")
    args = parser.parse_args(args)

    opts = {
        "print_file_list": args.ls_files,
        "print_corpus_dir": args.corpus_dir,
        "print_model_dir": args.model_dir,
        "print_sampler_dir": args.sampler_dir,
    }

    main(clgen.main, args.model, args.sampler, **opts)
