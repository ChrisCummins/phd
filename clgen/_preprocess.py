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
Preprocess OpenCL files for machine learning.
"""
import json
import labm8
import math
import os
import progressbar
import random
import re
import shutil
import sqlite3
import sys

from functools import partial
from io import open
from labm8 import fs
from multiprocessing import cpu_count, Pool
from queue import Queue
from queue import Empty as QueueEmpty
from subprocess import Popen, PIPE, STDOUT
from tempfile import NamedTemporaryFile
from threading import Thread
from typing import Dict, List, Tuple


import clgen
from clgen import dbutil
from clgen import log
from clgen import native


#
# Custom exceptions:
#

# Internal exceptions:
class LlvmException(clgen.CLgenError):
    """LLVM Error"""
    pass


class OptException(LlvmException):
    """
    LLVM opt error.
    """
    pass


class BadCodeException(clgen.CLgenError):
    """
    Code is bad.
    """
    pass


class ClangException(BadCodeException):
    """
    clang error.
    """
    pass


class ClangFormatException(BadCodeException):
    """
    clang-format error.
    """
    pass


class UglyCodeException(clgen.CLgenError):
    """
    Code is ugly.
    """
    pass


class InstructionCountException(UglyCodeException):
    """
    Instruction count error.
    """
    pass


class NoCodeException(UglyCodeException):
    """
    Sample contains no code.
    """
    pass


class RewriterException(UglyCodeException):
    """
    Program rewriter error.
    """
    pass


class GPUVerifyException(UglyCodeException):
    """
    GPUVerify found a bug.
    """
    pass


class GPUVerifyTimeoutException(GPUVerifyException):
    """
    GPUVerify timed out.
    """
    pass


CLANG_CL_TARGETS = [
    'nvptx64-nvidia-nvcl',
    'spir64'
]


def clang_cl_args(target: str=CLANG_CL_TARGETS[0],
                  use_shim: bool=True, error_limit: int=0) -> List[str]:
    """
    Get the Clang args to compile OpenCL.

    Parameters
    ----------
    target : str
        LLVM target.
    use_shim : bool, optional
        Inject shim header.
    error_limit : int, optional
        Limit number of compiler errors.

    Returns
    -------
    List[str]
        Array of args.
    """
    # clang warnings to disable
    disabled_warnings = [
        'ignored-pragmas',
        'implicit-function-declaration',
        'incompatible-library-redeclaration',
        'macro-redefined',
    ]

    args = [
        '-I' + fs.path(native.LIBCLC),
        '-target', target,
        '-ferror-limit={}'.format(error_limit),
        '-xcl'
    ] + ['-Wno-{}'.format(x) for x in disabled_warnings]

    if use_shim:
        args += ['-include', native.SHIMFILE]

    return args


def compiler_preprocess_cl(src: str, id: str='anon',
                           use_shim: bool=True) -> str:
    """
    Preprocess OpenCL file.

    Inlines macros, removes comments, etc.

    Parameters
    ----------
    src : str
        OpenCL source.
    id : str, optional
        Name of OpenCL source.
    use_shim : bool, optional
        Inject shim header.

    Returns
    -------
    str
        Preprocessed source.

    Raises
    ------
    ClangException
        If compiler errors.
    """
    cmd = [native.CLANG] + clang_cl_args(use_shim=use_shim) + [
        '-E', '-c', '-', '-o', '-'
    ]
    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate(src.encode('utf-8'))

    if process.returncode != 0:
        raise ClangException(stderr.decode('utf-8'))

    src = stdout.decode('utf-8')
    lines = src.split('\n')

    # Strip all the includes:
    for i, line in enumerate(lines):
        if line == '# 1 "<stdin>" 2':
            break
    src = '\n'.join(lines[i + 1:]).strip()

    # Strip lines beginning with '#' (that's preprocessor
    # stuff):
    src = '\n'.join([line for line in src.split('\n')
                     if not line.startswith('#')])

    return src


def rewrite_cl(src: str, id: str='anon', use_shim: bool=True) -> str:
    """
    Rewrite OpenCL sources.

    Renames all functions and variables with short, unique names.

    Parameters
    ----------
    src : str
        OpenCL source.
    id : str, optional
        OpenCL source name.
    use_shim : bool, optional
        Inject shim header.

    Returns
    -------
    str
        Rewritten OpenCL source.

    Raises
    ------
    RewriterException
        If rewriter fails.
    """
    # Rewriter can't read from stdin.
    with NamedTemporaryFile('w', suffix='.cl') as tmp:
        tmp.write(src)
        tmp.flush()
        cmd = ([native.CLGEN_REWRITER, tmp.name] +
               ['-extra-arg=' + x
                for x in clang_cl_args(use_shim=use_shim)] + ['--'])

        process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()

    # If there was nothing to rewrite, rewriter exits with error code:
    EUGLY_CODE = 204
    if process.returncode == EUGLY_CODE:
        # Propagate the error:
        raise RewriterException(src)
    # NOTE: the rewriter process can still fail because of some other
    # compilation problem, e.g. for some reason the 'enable 64bit
    # support' pragma which should be included in the shim isn't being
    # propogated correctly to the rewriter. However, the rewriter will
    # still correctly process the input, so we ignore all error codes
    # except the one we care about (EUGLY_CODE).
    return stdout.decode('utf-8')


def compile_cl_bytecode(src: str, id: str='anon', use_shim: bool=True) -> str:
    """
    Compile OpenCL kernel to LLVM bytecode.

    Parameters
    ----------
    src : str
        OpenCL source.
    id : str, optional
        Name of OpenCL source.
    use_shim : bool, optional
        Inject shim header.

    Returns
    -------
    str
        Bytecode.

    Raises
    ------
    ClangException
        If compiler errors.
    """
    cmd = [native.CLANG] + clang_cl_args(use_shim=use_shim) + [
        '-emit-llvm', '-S', '-c', '-', '-o', '-'
    ]

    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate(src.encode('utf-8'))

    if process.returncode != 0:
        raise ClangException(stderr.decode('utf-8'))
    return stdout


def gpuverify(src: str, args: list, id: str='anon', timeout=60) -> str:
    """
    Run GPUverify over kernel.

    Parameters
    ----------
    src : str
        OpenCL source.
    id : str, optional
        OpenCL source name.

    Returns
    -------
    str
        OpenCL source.

    Raises
    ------
    GPUVerifyException
        If GPUverify finds a bug.
    InternalError
        If GPUverify fails.
    """
    # FIXME: GPUVerify support on macOS.
    from labm8 import system
    if not system.is_linux():
        raise clgen.InternalError("GPUVerify only supported on Linux!")

    # GPUverify can't read from stdin.
    with NamedTemporaryFile('w', suffix='.cl') as tmp:
        tmp.write(src)
        tmp.flush()
        cmd = ['timeout', '-s9', str(timeout), native.GPUVERIFY, tmp.name] + args

        process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()

    if process.returncode == -9:  # timeout signal
        raise GPUVerifyTimeoutException(f"GPUveryify failed to complete with {timeout} seconds")
    elif process.returncode != 0:
        raise GPUVerifyException(stderr.decode('utf-8'))

    return src


_instcount_re = re.compile(
    r"^(?P<count>\d+) instcount - Number of (?P<type>.+)")


def parse_instcounts(txt: str) -> Dict[str, int]:
    """
    Parse LLVM opt instruction counts pass.

    Parameters
    ----------
    txt : str
        LLVM output.

    Returns
    -------
    Dict[str, int]
        key, value pairs, where key is instruction type and value is
        instruction type count.
    """
    lines = [x.strip() for x in txt.split("\n")]
    counts = {}

    # Build a list of counts for each type.
    for line in lines:
        match = re.search(_instcount_re, line)
        if match:
            count = int(match.group("count"))
            key = match.group("type")
            if key in counts:
                counts[key].append(count)
            else:
                counts[key] = [count]

    # Sum all counts.
    for key in counts:
        counts[key] = sum(counts[key])

    return counts


def instcounts2ratios(counts: Dict[str, int]) -> Dict[str, float]:
    """
    Convert instruction counts to instruction densities.

    If, for example, there are 10 instructions and 2 addition instructions,
    then the instruction density of add operations is .2.

    Parameters
    ----------
    counts : Dict[str, int]
        Key value pairs of instruction types and counts.

    Returns
    -------
    ratios : Dict[str, float]
        Key value pairs of instruction types and densities.
    """
    if not len(counts):
        return {}

    ratios = {}
    total_key = "instructions (of all types)"
    non_ratio_keys = [
        total_key
    ]
    total = float(counts[total_key])

    for key in non_ratio_keys:
        ratios[dbutil.escape_sql_key(key)] = counts[key]

    for key in counts:
        if key not in non_ratio_keys:
            # Copy count
            ratios[dbutil.escape_sql_key(key)] = counts[key]
            # Insert ratio
            ratio = float(counts[key]) / total
            ratios[dbutil.escape_sql_key('ratio_' + key)] = ratio

    return ratios


def bytecode_features(bc: str, id: str='anon') -> Dict[str, float]:
    """
    Extract features from bytecode.

    Parameters
    ----------
    bc : str
        LLVM bytecode.
    id : str, optional
        Name of OpenCL source.

    Returns
    -------
    Dict[str, float]
        Key value pairs of instruction types and densities.

    Raises
    ------
    OptException
        If LLVM opt pass errors.
    """
    cmd = [native.OPT, '-analyze', '-stats', '-instcount', '-']

    # LLVM pass output pritns to stderr, so we'll pipe stderr to
    # stdout.
    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    stdout, _ = process.communicate(bc)

    if process.returncode != 0:
        raise OptException(stdout.decode('utf-8'))

    instcounts = parse_instcounts(stdout.decode('utf-8'))
    instratios = instcounts2ratios(instcounts)

    return instratios

# Options to pass to clang-format.
#
# See: http://clang.llvm.org/docs/ClangFormatStyleOptions.html
#
clangformat_config = {
    'BasedOnStyle': 'Google',
    'ColumnLimit': 500,
    'IndentWidth': 2,
    'AllowShortBlocksOnASingleLine': False,
    'AllowShortCaseLabelsOnASingleLine': False,
    'AllowShortFunctionsOnASingleLine': False,
    'AllowShortLoopsOnASingleLine': False,
    'AllowShortIfStatementsOnASingleLine': False,
    'DerivePointerAlignment': False,
    'PointerAlignment': 'Left'
}


def clangformat_ocl(src: str, id: str='anon') -> str:
    """
    Enforce code style on OpenCL file.

    Parameters
    ----------
    src : str
        OpenCL source.
    id : str, optional
        Name of OpenCL source.

    Returns
    -------
    str
        Styled source.

    Raises
    ------
    ClangFormatException
        If formatting errors.
    """
    cmd = [native.CLANG_FORMAT, '-style={}'.format(
        json.dumps(clangformat_config))]
    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate(src.encode('utf-8'))

    if stderr:
        log.error(stderr.decode('utf-8'))
    if process.returncode != 0:
        raise ClangFormatException(stderr.decode('utf-8'))

    return stdout.decode('utf-8')


def verify_bytecode_features(bc_features: Dict[str, float],
                             id: str='anon') -> None:
    """
    Verify LLVM bytecode features.

    Parameters
    ----------
    bc_features : Dict[str, float]
        Bytecode features.
    id : str, optional
        Name of OpenCL source.

    Raises
    ------
    InstructionCountException
        If verification errors.
    """
    # The minimum number of instructions before a kernel is discarded
    # as ugly.
    min_num_instructions = 1
    num_instructions = bc_features.get('instructions_of_all_types', 0)

    if num_instructions < min_num_instructions:
        raise InstructionCountException(
            'Code contains {} instructions. The minimum allowed is {}'
            .format(num_instructions, min_num_instructions))


def ensure_has_code(src: str) -> str:
    """
    Check that file contains actual executable code.

    Parameters
    ----------
    src : str
        OpenCL source, must be preprocessed.

    Returns
    -------
    src
        Unmodified source code.

    Raises
    ------
    NoCodeException
        If kernel is empty.
    """
    if len(src.split('\n')) < 3:
        raise NoCodeException

    return src


def sanitize_prototype(src: str) -> str:
    """
    Sanitize OpenCL prototype.

    Ensures that OpenCL prototype fits on a single line.

    Parameters
    ----------
    src : str
        OpenCL source.

    Returns
    -------
    src
        Source code with sanitized prototypes.

    Returns
    -------
    str
        Sanitized OpenCL source.
    """
    # Ensure that prototype is well-formed on a single line:
    try:
        prototype_end_idx = src.index('{') + 1
        prototype = ' '.join(src[:prototype_end_idx].split())
        return prototype + src[prototype_end_idx:]
    except ValueError:
        # Ok so erm... if the '{' character isn't found, a ValueError
        # is thrown. Why would '{' not be found? Who knows, but
        # whatever, if the source file got this far through the
        # preprocessing pipeline then it's probably "good" code. It
        # could just be that an empty file slips through the cracks or
        # something.
        return src


def preprocess_opencl(src: str, id: str='anon', use_shim: bool=True,
                      use_gpuverify: bool=False) -> str:
    """
    Preprocess an OpenCL source. There are three possible outcomes:

    1. Good. Code is preprocessed and ready to be put into a training set.
    2. Bad. Code can't be preprocessed (i.e. it's "bad" OpenCL).
    3. Ugly. Code can be preprocessed but isn't useful for training
       (e.g. it's an empty file).

    Parameters
    ----------
    src : str
        The source code as a string.
    id : str, optional
        An identifying name for the source code (used in exception messages).
    use_shim : bool, optional
        Inject shim header.
    use_gpuverify : bool, optional
        Whether to run GPUVerify on the code.

    Returns
    -------
    str
        Preprocessed source code as a string.

    Raises
    ------
    BadCodeException
        If code is bad (see above).
    UglyCodeException
        If code is ugly (see above).
    clgen.InternalException
        In case of some other error.
    """
    # Compile to bytecode and verify features:
    bc = compile_cl_bytecode(src, id, use_shim)
    bc_features = bytecode_features(bc, id)
    verify_bytecode_features(bc_features, id)

    # Rewrite and format source:
    src = compiler_preprocess_cl(src, id, use_shim)
    src = rewrite_cl(src, id, use_shim)
    src = clangformat_ocl(src, id).strip()
    src = ensure_has_code(src)
    src = sanitize_prototype(src)

    if use_gpuverify:
        gpuverify(src)

    return src


def _strip_comments(text: str):
    """
    Strip C/C++ style comments.

    written by @markus-jarderot https://stackoverflow.com/a/241506/1318051
    """
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def _remove_duplicate_empty_lines(text: str):
    """
    Truncate blank lines.
    """
    last_line = None
    lines = []
    for line in text.split("\n"):
        line = line.rstrip()
        if line or last_line:
            lines.append(line)
        last_line = line
    return "\n".join(lines)


def preprocess_solidity(src: str, id: str='anon', **kwargs) -> str:
    """
    Preprocess a solidity source.
    """
    src = _strip_comments(src)
    src = _remove_duplicate_empty_lines(src)
    src = clangformat_ocl(src)
    return src


def preprocess(src: str, id: str="anon", lang: str="opencl",
               **lang_opts) -> str:
    """
    Preprocess a file. There are three possible outcomes:

    1. Good. Code is preprocessed and ready to be put into a training set.
    2. Bad. Code can't be preprocessed (i.e. it's "bad" code).
    3. Ugly. Code can be preprocessed but isn't useful for training
       (e.g. it's an empty file).

    Parameters
    ----------
    src : str
        The source code as a string.
    id : str, optional
        An identifying name for the source code (used in exception messages).

    Returns
    -------
    str
        Preprocessed source code as a string.

    Raises
    ------
    BadCodeException
        If code is bad (see above).
    UglyCodeException
        If code is ugly (see above).
    clgen.InternalException
        In case of some other error.
    """
    if lang == "opencl":
        return preprocess_opencl(src, id, **lang_opts)
    elif lang == "solidity":
        return preprocess_solidity(src, id, **lang_opts)
    else:
        raise ValueError(f"unsuporrted language '{lang}'")


def preprocess_for_db(src: str, **preprocess_opts) -> Tuple[int, str]:
    """
    Preprocess source code for import into contentdb.

    Parameters
    ----------
    src : str
        Source to preprocess.
    **preprocess_opts
        Preprocessing options.

    Returns
    -------
    Tuple[int, str]
        The status of the preprocessed code, and the preprocess output.
    """
    try:
        # Try and preprocess it:
        status = 0
        contents = preprocess(src, **preprocess_opts)
    except BadCodeException as e:
        status = 1
        contents = str(e)
    except UglyCodeException as e:
        status = 2
        contents = str(e)

    return status, contents


def preprocess_file(path: str, inplace: bool=False, **preprocess_opts) -> None:
    """
    Preprocess a file.

    Prints output to stdout by default. If preprocessing fails, this function
    exits.

    Parameters
    ----------
    path : str
        String path to file.
    inplace : bool, optional
        If True, overwrite input file.
    """
    with open(path) as infile:
        contents = infile.read()
    try:
        out = preprocess(contents, **preprocess_opts)
        if inplace:
            with open(path, 'w') as outfile:
                outfile.write(out)
        else:
            print(out)
    except BadCodeException as e:
        log.fatal(e, ret=1)
    except UglyCodeException as e:
        log.fatal(e, ret=2)


def _preprocess_inplace_worker(path: str) -> None:
    """worker function for preprocess_inplace()"""
    log.info('preprocess', path)
    preprocess_file(path, inplace=True)


def preprocess_inplace(paths: List[str], max_num_workers: int=cpu_count(),
                       max_attempts: int=100, attempt: int=1) -> None:
    """
    Preprocess a list of files in place.

    Parameters
    ----------
    paths : List[str]
        List of paths.
    max_num_workers : int, optional
        Number of processes to spawn.
    max_attempts : int, optional
        In case of an OSError or TimeoutError, this number of attempts will be
        made.
    """
    if attempt > max_attempts:
        raise clgen.InternalError(
            f"Failed to process files after {max_attempts} attempts")
    elif attempt > 1:
        log.warning("preprocess attempt #.", attempt)

    num_workers = min(len(paths), max_num_workers)

    try:
        log.info('spawned', num_workers, 'worker threads to process',
                 len(paths), 'files ...')
        with clgen.terminating(Pool(num_workers)) as pool:
            pool.map(_preprocess_inplace_worker, paths)
    except (OSError, TimeoutError) as e:
        log.error(e)

        # Try again with fewer threads.
        # See: https://github.com/ChrisCummins/clgen/issues/64
        max_num_workers = max(int(max_num_workers / 2), 1)
        preprocess_inplace(paths, max_num_workers=max_num_workers,
                           attempt=attempt + 1, max_attempts=max_attempts)


class PreprocessWorker(Thread):
    """ preprocessor worker thread """

    def __init__(self, jobs: List[Dict], queue: Queue):
        super(PreprocessWorker, self).__init__()
        self.jobs = jobs
        self.queue = queue

    def run(self):
        while self.jobs:
            job = self.jobs.pop(0)

            kid = job["id"]
            src = job["src"]
            preprocess_opts = job["preprocess_opts"]

            status, contents = preprocess_for_db(src, id=id, **preprocess_opts)

            result = {
                "id": kid,
                "status": status,
                "contents": contents
            }

            self.queue.put(result)


def _preprocess_db(db_path: str, max_num_workers: int=cpu_count(),
                   max_attempts: int=100, attempt: int=1,
                   **preprocess_opts) -> None:
    """
    Preprocess OpenCL dataset.

    Parameters
    ----------
    db_path : str
        OpenCL kernels dataset.
    max_num_workers : int, optional
        Number of processes to spawn.
    max_attempts : int, optional
        In case of an OSError or TimeoutError, this number of attempts will be
        made.
    """
    if attempt > max_attempts:
        raise clgen.InternalError(
            f"failed to preprocess files after {max_attempts} attempts")

    log.verbose("determining jobs")

    contentfiles = set(dbutil.kernel_ids(db_path, "ContentFiles"))
    preprocessedfiles = set(dbutil.kernel_ids(db_path, "PreprocessedFiles"))

    ncontentfiles = len(contentfiles)
    npreprocessedfiles = len(preprocessedfiles)

    todo = contentfiles - preprocessedfiles
    ntodo = len(todo)

    # check we have something to do
    if not ntodo:
        return

    todo_ratio = ntodo / ncontentfiles

    log.info("{ntodo} ({todo_ratio:.1%}) samples need preprocessing"
             .format(**vars()))

    log.verbose("creating jobs")

    # Determine if we need to inline kernels when creating jobs
    db = sqlite3.connect(db_path)
    c = db.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ContentMeta';")
    meta_table = c.fetchone()
    c.close()
    db.close()
    if meta_table:
        get_kernel = lambda kid: dbutil.get_inlined_kernel(db_path, kid, lang=preprocess_opts["lang"])
    else:
        get_kernel = lambda kid: dbutil.get_kernel(db_path, kid, table="ContentFiles")

    # create jobs
    jobs = [{
        "id": kid,
        "src": get_kernel(kid),
        "preprocess_opts": preprocess_opts,
    } for kid in todo]

    random.shuffle(jobs)

    # split size
    worker_njobs = math.ceil(ntodo / max_num_workers)

    # producer-consumer queue
    queue = Queue(maxsize=128)

    log.verbose(f"assigning {ntodo} jobs to {max_num_workers} threads")

    try:
        # our worker threads. these busy little bees will do the heavy lifting
        # of preprocessing the contentfiles, pushing their results onto
        # the queue
        producers = [PreprocessWorker(jobs[i:i+worker_njobs], queue)
                     for i in range(0, ntodo, worker_njobs)]

        # fly, my pretties, fly!
        for producer in producers:
            producer.start()

        # consume the results from the worker threads from the main thread
        for i in progressbar.ProgressBar()(range(ntodo)):
            # pull a fresh result from the queue (block if necessary)
            try:
                result = queue.get(timeout=60)
            except QueueEmpty as e:
                raise TimeoutError(
                    'failed to fetch result after 60 seconds. '
                    'something went wrong') from e

            # insert result into database
            db = dbutil.connect(db_path)
            c = db.cursor()
            c.execute("INSERT INTO PreprocessedFiles VALUES(?,?,?)",
                      (result["id"], result["status"], result["contents"]))
            c.close()
            db.commit()
            db.close()

        for producer in producers:
            producer.join()

    except (OSError, TimeoutError) as e:
        log.error(e)

        if attempt > 2 and not i:
            log.warning("no progress has been made since previous attempt. "
                        "I'm not going to try another attempt.")
            return


        # Try again with fewer threads.
        # See: https://github.com/ChrisCummins/clgen/issues/64
        max_num_workers = max(int(max_num_workers / 2), 1)
        _preprocess_db(db_path, max_num_workers=max_num_workers,
                       attempt=attempt + 1, max_attempts=max_attempts,
                       **preprocess_opts)


def preprocess_db(db_path: str, **preprocess_opts) -> bool:
    """
    Preprocess database contents.

    Parameters
    ----------
    db_path : str
        Path to database.

    Returns
    -------
    bool
        True if modified, false if no work needed.
    """
    db = dbutil.connect(db_path)

    modified = dbutil.is_modified(db)
    if modified:
        _preprocess_db(db_path, **preprocess_opts)
        dbutil.set_modified_status(db, modified)
        return True
    else:
        return False
