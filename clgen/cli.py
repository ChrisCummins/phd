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
import argparse
import cProfile
import inspect
import os
import sys
import traceback

from argparse import RawTextHelpFormatter
from labm8 import jsonutil, fs, prof, types
from sys import exit
from typing import List

import clgen
from clgen import log
from clgen import dbutil

__help_epilog__ = """
Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
<http://chriscummins.cc/clgen>
"""

def getself(func):
    """ decorator to pass function as first argument to function """
    def wrapper(*args, **kwargs):
        return func(func, *args, **kwargs)
    return wrapper


def run(method, *args, **kwargs):
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
        def runctx():
            method(*args, **kwargs)

        if prof.is_enabled() and log.is_verbose():
            return cProfile.runctx('runctx()', None, locals(), sort='tottime')
        else:
            return runctx()
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


def _version():
    """ print version and quit """
    version = clgen.version()
    print(f"clgen {version} made with \033[1;31m♥\033[0;0m by "
          "Chris Cummins <chrisc.101@gmail.com>.")


@getself
def _test(self, args):
    """
    Run the CLgen self-test suite.
    """
    import clgen.test  # Must scope this import, as it breaks cache behaviour

    if args.coveragerc_path:
        print(clgen.test.coveragerc_path())
        sys.exit(0)

    if args.coverage_path:
        print(clgen.test.coverage_report_path())
        sys.exit(0)

    sys.exit(clgen.test.testsuite())


@getself
def _train(self, args):
    """
    Train a CLgen model.
    """
    model_json = jsonutil.loads(args.model.read())
    model = clgen.Model.from_json(model_json)
    model.train()
    log.info("done.")


@getself
def _sample(self, args) -> None:
    """
    Sample a model.
    """
    model_json = jsonutil.loads(args.model.read())
    model = clgen.Model.from_json(model_json)

    sampler_json = jsonutil.loads(args.sampler.read())
    sampler = clgen.Sampler.from_json(sampler_json)

    model.train()
    sampler.sample(model)


@getself
def _fetch(self, args):
    """
    Import OpenCL files into kernel datbase.

    The kernel database is used as a staging ground for input files, which are
    then preprocessed and assembled into corpuses. This program acts as the front
    end, assembling files from the file system into a database for preprocessing.
    """
    db_path = os.path.expanduser(args.input)

    clgen.fetch(db_path, args.paths)
    log.info("done.")


@getself
def _fetch_github(self, args):
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

    db_path = args.input

    try:
        github_username = environ['GITHUB_USERNAME']
        github_pw = environ['GITHUB_PW']
        github_token = environ['GITHUB_TOKEN']
    except KeyError as e:
        log.fatal('environment variable {} not set'.format(e))

    try:
        fetch_github(db_path, github_username, github_pw, github_token)
    except BadCredentialsException as e:
        log.fatal("bad GitHub credentials")


@getself
def _ls_files(self, args):
    """
    Import OpenCL files into kernel datbase.

    The kernel database is used as a staging ground for input files, which are
    then preprocessed and assembled into corpuses. This program acts as the front
    end, assembling files from the file system into a database for preprocessing.
    """
    model_json = jsonutil.loads(args.model.read())
    model = clgen.Model.from_json(model_json)

    caches = [model.corpus.cache, model.cache]

    if args.sampler:
        sampler_json = jsonutil.loads(args.sampler.read())
        sampler = clgen.Sampler.from_json(sampler_json)
        caches.append(sampler.cache(model))

    files = sorted(
        types.flatten(c.ls(abspaths=True, recursive=True) for c in caches))
    print('\n'.join(files))


@getself
def _ls_models(self, args):
    """
    List all locally cached models.
    """
    print(clgen.models_to_tab(*clgen.models()))


@getself
def _ls_samplers(self, args):
    """
    List all locally cached samplers.
    """
    log.warning("not implemented")


@getself
def _db_init(self, args):
    """
    Create an empty OpenCL kernel database.
    """
    dbutil.create_db(args.input, args.github)
    print(fs.abspath(args.input))


@getself
def _db_explore(self, args):
    """
    Exploratory analysis of preprocessed dataset.

    Provides an overview of the contents of an OpenCL kernel database.
    """
    clgen.explore(args.input)


@getself
def _db_merge(self, args):
    """
    Merge kernel datasets.
    """
    dbutil.merge(args.dataset, args.inputs)


@getself
def _preprocess(self, args):
    """
    Process OpenCL files for machine learning.

    This is a three step process. First, the OpenCL kernels are compiled to
    bytecode, then the source files are preprocessed, before being rewritten.

    Preprocessing is computationally demanding and highly paralellised.
    Expect high resource contention during preprocessing.
    """
    if args.file and args.inplace:
        clgen.preprocess_inplace(args.inputs, use_gpuverify=args.gpuverify)
    else:
        for path in args.inputs:
            if args.file:
                clgen.preprocess_file(path, inplace=False,
                                      use_gpuverify=args.gpuverify)
            elif args.remove_bad_preprocessed:
                dbutil.remove_bad_preprocessed(path)
            elif args.remove_preprocessed:
                dbutil.remove_preprocessed(path)
                print("done.")
            else:
                if clgen.preprocess_db(path, use_gpuverify=args.gpuverify):
                    print("done.")
                else:
                    print("nothing to be done.")


@getself
def _db_dump(self, args):
    """
    Dump kernel dataset to file(s).
    """
    opts = {
        "dir": args.d,
        "eof": args.eof,
        "fileid": args.i,
        "input_samples": args.input_samples,
        "reverse": args.r,
        "status": args.status
    }

    dbutil.dump_db(args.input, args.out_path, **opts)


@getself
def _features(self, args):
    """
    Extract static OpenCL kernel features.

    This extracts the static compile-time features of the paper:

        Grewe, D., Wang, Z., & O'Boyle, M. F. P. M. (2013). Portable Mapping of
        Data Parallel Programs to OpenCL for Heterogeneous Systems. In CGO. IEEE.
    """
    from clgen import features

    def features_dir(csv_path):
        return fs.basename(fs.dirname(csv_path))

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

    features.files(paths, fatal_errors=args.fatal_errors,
                   use_shim=args.shim, quiet=args.quiet,
                   header=not args.no_header)


@getself
def _atomize(self, args):
    """
    Extract and print corpus vocabulary.
    """
    with open(clgen.must_exist(args.input)) as infile:
        data = infile.read()
    atoms = corpus.atomize(data, vocab=args.type)

    if args.size:
        log.info("size:", len(atoms))
    else:
        log.info('\n'.join(atoms))


@getself
def _cache_migrate(self, args):
    """
    Refresh the cached model, corpus, and sampler IDs.
    """
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


@getself
def main(self, args: List[str]=sys.argv[1:]):
    """
    A deep learning program generator for the OpenCL programming language.

    The core operations of CLgen are:

       1. OpenCL files are collected from a model specification file.
       2. These files are preprocessed into an OpenCL kernel database.
       3. A training corpus is generated from the input files.
       4. A machine learning model is trained on the corpus of files.
       5. The trained model is sampled for new kernels.
       6. The samples are tested for compilability.

    This program automates the execution of all six stages of the pipeline.
    The pipeline can be interrupted and resumed at any time. Results are cached
    across runs. If installed with CUDA support, NVIDIA GPUs will be used to
    improve performance where possible.
    """
    parser = argparse.ArgumentParser(
        prog="clgen",
        description=inspect.getdoc(self),
        epilog="""
For information about a specific command, run `clgen <command> --help`.

""" + __help_epilog__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="increase output verbosity")
    parser.add_argument(
        "--version", action="store_true",
        help="show version information and exit")
    parser.add_argument(
        "--debug", action="store_true",
        help="in case of error, print debugging information")
    parser.add_argument(
        "--profile", action="store_true",
        help=("enable internal API profiling. When combined with --verbose, "
              "prints a complete profiling trace"))

    parser.add_argument(
        "--corpus-dir", metavar="<corpus>",
        type=argparse.FileType("r"),
        help="print path to corpus cache")
    parser.add_argument(
        "--model-dir", metavar="<model>",
        type=argparse.FileType("r"),
        help="print path to model cache")
    parser.add_argument(
        "--sampler-dir", metavar=("<model>", "<sampler>"),
        type=argparse.FileType("r"), nargs=2,
        help="print path to sampler cache")

    subparser = parser.add_subparsers(title="available commands")

    # test
    test = subparser.add_parser("test", help="run the testsuite",
                                description=inspect.getdoc(_test),
                                epilog=__help_epilog__)
    test.set_defaults(dispatch_func=_test)
    group = test.add_mutually_exclusive_group()
    group.add_argument("--coveragerc-path", action="store_true",
                       help="print path to coveragerc file")
    group.add_argument("--coverage-path", action="store_true",
                       help="print path to coverage file")

    # train
    train = subparser.add_parser("train", aliases=["t", "tr"],
                                 help="train models",
                                 description=inspect.getdoc(_train),
                                 epilog=__help_epilog__)
    train.set_defaults(dispatch_func=_train)
    train.add_argument("model", metavar="<model>",
                       type=argparse.FileType("r"),
                       help="path to model specification file")

    # sample
    sample = subparser.add_parser("sample", aliases=["s", "sa"],
                                  help="train and sample models",
                                  description=inspect.getdoc(_sample),
                                  epilog=__help_epilog__)
    sample.set_defaults(dispatch_func=_sample)
    sample.add_argument("model", metavar="<model>",
                        type=argparse.FileType("r"),
                        help="path to model specification file")
    sample.add_argument("sampler", metavar="<sampler>",
                        type=argparse.FileType("r"),
                        help="path to sampler specification file")

    # fetch
    fetch = subparser.add_parser("fetch", aliases=["f", "fe"],
                                 help="gather training data",
                                 description="Fetch OpenCL kernels",
                                 epilog=__help_epilog__)
    fetch_parser = fetch.add_subparsers(title="available commands")

    fetch_fs = fetch_parser.add_parser("fs", help="fetch from filesystem",
                                       description=inspect.getdoc(_fetch),
                                       epilog=__help_epilog__)
    fetch_fs.set_defaults(dispatch_func=_fetch)
    fetch_fs.add_argument('input', metavar="<db>", type=str,
                          help='path to SQL dataset')
    fetch_fs.add_argument('paths', metavar="<path>", nargs='+', type=str,
                          help='path to OpenCL files or directories')

    fetch_gh = fetch_parser.add_parser("github", help="mine OpenCL from GitHub",
                                       description=inspect.getdoc(_fetch_github),
                                       epilog=__help_epilog__)
    fetch_gh.set_defaults(dispatch_func=_fetch_github)
    fetch_gh.add_argument('input', metavar="<db>", type=str,
                         help='path to SQL dataset')

    # ls
    ls = subparser.add_parser("ls",
                              help="list files",
                              description="list files",
                              epilog=__help_epilog__)
    ls_parser = ls.add_subparsers(title="available commands")

    ls_files = ls_parser.add_parser("files", help="list cached files",
                                    description=inspect.getdoc(_ls_files),
                                    epilog=__help_epilog__)
    ls_files.set_defaults(dispatch_func=_ls_files)
    ls_files.add_argument("model", metavar="<model>",
                          type=argparse.FileType("r"),
                          help="path to model specification file")
    ls_files.add_argument("sampler", metavar="<sampler>", nargs="?",
                          type=argparse.FileType("r"),
                          help="path to sampler specification file")

    ls_models = ls_parser.add_parser("models", help="list cached models",
                                     description=inspect.getdoc(_ls_models),
                                     epilog=__help_epilog__)
    ls_models.set_defaults(dispatch_func=_ls_models)

    ls_samplers = ls_parser.add_parser("samplers", help="list cached samplers",
                                       description=inspect.getdoc(_ls_samplers),
                                       epilog=__help_epilog__)
    ls_samplers.set_defaults(dispatch_func=_ls_samplers)

    # db
    db = subparser.add_parser("db",
                              help="manage databases",
                              description="manage databases",
                              epilog=__help_epilog__)
    db_parser = db.add_subparsers(title="available commands")

    db_init = db_parser.add_parser("init", help="create a database",
                                   description=inspect.getdoc(_db_init),
                                   epilog=__help_epilog__)
    db_init.set_defaults(dispatch_func=_db_init)
    db_init.add_argument('input', metavar="<db>",
                         help='path to SQL dataset')
    db_init.add_argument('-g', '--github', action='store_true',
                         help='generate dataset with GitHub metadata')

    db_explore = db_parser.add_parser("explore", help="show database stats",
                                      description=inspect.getdoc(_db_explore),
                                      epilog=__help_epilog__)
    db_explore.set_defaults(dispatch_func=_db_explore)
    db_explore.add_argument('input', metavar="<db>",
                            help='path to SQL dataset')

    db_merge = db_parser.add_parser("explore", help="show database stats",
                                    description=inspect.getdoc(_db_merge),
                                    epilog=__help_epilog__)
    db_merge.set_defaults(dispatch_func=_db_merge)
    db_merge.add_argument("dataset", metavar="<db>", help="path to output dataset")
    db_merge.add_argument("inputs", metavar="<db>", nargs='+',
                          help="path to input datasets")


    db_dump = db_parser.add_parser("dbump", help="export database contents",
                                   description=inspect.getdoc(_db_dump),
                                   epilog=__help_epilog__)
    db_dump.set_defaults(dispatch_func=_db_merge)
    db_dump.add_argument('input', metavar="<db>",
                         help='path to kernels database')
    db_dump.add_argument('output', metavar="<path>",
                         help='path to output file or directory')
    db_dump.add_argument("-d", "--dir", action='store_true',
                         help='output to directory (overrides -i, --eof, -r)')
    db_dump.add_argument("-i", "--file-sep", action='store_true',
                         help='include file separators')
    db_dump.add_argument('--input-samples', action='store_true',
                         help='use input contents, not preprocessed')
    db_dump.add_argument('--eof', action='store_true', default=False,
                         help='print end of file')
    db_dump.add_argument('-r', action='store_true', default=False,
                         help='use reverse order')
    db_dump.add_argument('-s', '--status', type=int, default=0,
                         help='status code to use')


    # preprocess
    preprocess = subparser.add_parser(
        "preprocess", aliases=["p", "pp"],
        help="preprocess files for training",
        description=inspect.getdoc(_preprocess),
        epilog=__help_epilog__)
    preprocess.set_defaults(dispatch_func=_preprocess)
    preprocess.add_argument('inputs', metavar="<path>", nargs='+',
                            help='path to input')
    preprocess.add_argument('-f', '--file', action='store_true',
                            help='treat input as file')
    preprocess.add_argument('-i', '--inplace', action='store_true',
                            help='inplace file rewrite')
    preprocess.add_argument('-G', '--gpuverify', action='store_true',
                            help='run GPUVerify on kernels')
    group = preprocess.add_mutually_exclusive_group()
    group.add_argument('--remove-bad-preprocessed', action='store_true',
                       help="""\
delete the contents of all bad or ugly preprocessed files,
but keep the entries in the table""")
    group.add_argument("--remove-preprocessed", action="store_true",
                       help="remove all preprocessed files from database")

    # features
    features = subparser.add_parser(
        "features",
        help="get kernel features",
        description=inspect.getdoc(_features),
        epilog=__help_epilog__)
    features.set_defaults(dispatch_func=_features)
    features.add_argument("inputs", metavar="<path>", nargs="+",
                         help="input path(s)")
    features.add_argument("-d", "--dir", action="store_true",
                         help="treat inputs as directories")
    features.add_argument("-s", "--stats", action="store_true",
                         help="summarize a features files")
    features.add_argument("-e", "--fatal-errors", action="store_true",
                         help="quit on compiler error")
    features.add_argument("--shim", action="store_true",
                         help="include shim header")
    features.add_argument("-q", "--quiet", action="store_true",
                         help="minimal error output")
    features.add_argument("-H", "--no-header", action="store_true",
                         help="no features header")

    # atomize
    atomize = subparser.add_parser("atomize", help="atomize files",
                                   description=inspect.getdoc(_atomize),
                                   epilog=__help_epilog__)
    atomize.set_defaults(dispatch_func=_atomize)
    atomize.add_argument('input', help='path to input text file')
    atomize.add_argument('-t', '--type', type=str, default='char',
                         help='vocabulary type')
    atomize.add_argument('-s', '--size', action="store_true",
                         help="print vocabulary size")

    # cache
    cache = subparser.add_parser("cache",
                                 help="manage filesystem cache",
                                 description="manage filesystem cache",
                                 epilog=__help_epilog__)
    cache_parser = cache.add_subparsers(title="available commands")

    cache_migrate = cache_parser.add_parser(
        "migrate",
        help="migrate the cache",
        description=inspect.getdoc(_cache_migrate),
        epilog=__help_epilog__)
    cache_migrate.set_defaults(dispatch_func=_cache_migrate)

    args = parser.parse_args(args)

    # set log level
    log.init(args.verbose)

    # set debug option
    if args.debug:
        os.environ["DEBUG"] = "1"

    # set profile option
    if args.profile:
        prof.enable()

    # options whch override the normal argument parsing process.
    if args.version:
        _version()
    elif args.corpus_dir:
        model = clgen.Model.from_json(jsonutil.loads(args.corpus_dir.read()))
        print(model.corpus.cache.path)
    elif args.model_dir:
        model = clgen.Model.from_json(jsonutil.loads(args.model_dir.read()))
        print(model.cache.path)
    elif args.sampler_dir:
        model = clgen.Model.from_json(jsonutil.loads(args.sampler_dir[0].read()))
        sampler = clgen.Sampler.from_json(jsonutil.loads(args.sampler_dir[1].read()))
        print(sampler.cache(model).path)
        sys.exit(0)
    else:
        run(args.dispatch_func, args)
