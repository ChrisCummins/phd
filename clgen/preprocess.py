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
import re
import shutil
import sqlite3
import sys

from functools import partial
from io import open
from labm8 import fs
from multiprocessing import cpu_count, Pool
from subprocess import Popen, PIPE, STDOUT
from tempfile import NamedTemporaryFile


import clgen
from clgen import clutil
from clgen import dbutil
from clgen import log
from clgen import native
from clgen.cache import Cache


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


CLANG_CL_TARGETS = [
    'nvptx64-nvidia-nvcl',
    'spir64'
]


def clang_cl_args(target: str=CLANG_CL_TARGETS[0],
                  use_shim: bool=True, error_limit: int=0) -> list:
    """
    Get the Clang args to compile OpenCL.

    Arguments:
        target (str): LLVM target.
        use_shim (bool, optional): Inject shim header.
        error_limit (int, optional): Limit number of compiler errors.

    Returns:
        str[]: Array of args.
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

    Arguments:
        src (str): OpenCL source.
        id (str, optional): Name of OpenCL source.
        use_shim (bool, optional): Inject shim header.

    Returns:
        str: Preprocessed source.

    Raises:
        ClangException: If compiler errors.
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


def gpuverify(src: str, args: list, id: str='anon') -> str:
    """
    Run GPUverify over kernel.

    Arguments:
        src (str): OpenCL source.
        id (str, optional): OpenCL source name.

    Returns:
        str: OpenCL source.

    Raises:
        GPUVerifyException: If GPUverify finds a bug.
        InternalError: If GPUverify fails.
    """
    # GPUverify can't read from stdin.
    with NamedTemporaryFile('w', suffix='.cl') as tmp:
        tmp.write(src)
        tmp.flush()
        cmd = ([native.GPUVERIFY, tmp.name] + args)

        process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise GPUVerifyException(stderr.decode('utf-8'))

    return src


def rewrite_cl(src: str, id: str='anon', use_shim: bool=True) -> str:
    """
    Rewrite OpenCL sources.

    Renames all functions and variables with short, unique names.

    Arguments:
        src (str): OpenCL source.
        id (str, optional): OpenCL source name.
        use_shim (bool, optional): Inject shim header.

    Returns:
        str: Rewritten OpenCL source.

    Raises:
        RewriterException: If rewriter fails.
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
    rewritten = stdout.decode('utf-8')

    # Remove __attribute__ qualifiers
    stripped = clutil.strip_attributes(rewritten)

    return stripped


def compile_cl_bytecode(src: str, id: str='anon', use_shim: bool=True) -> str:
    """
    Compile OpenCL kernel to LLVM bytecode.

    Arguments:
        src (str): OpenCL source.
        id (str, optional): Name of OpenCL source.
        use_shim (bool, optional): Inject shim header.

    Returns:
        str: Bytecode.

    Raises:
        ClangException: If compiler errors.
    """
    cmd = [native.CLANG] + clang_cl_args(use_shim=use_shim) + [
        '-emit-llvm', '-S', '-c', '-', '-o', '-'
    ]

    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate(src.encode('utf-8'))

    if process.returncode != 0:
        raise ClangException(stderr.decode('utf-8'))
    return stdout


_instcount_re = re.compile(
    r"^(?P<count>\d+) instcount - Number of (?P<type>.+)")


def parse_instcounts(txt: str) -> dict:
    """
    Parse LLVM opt instruction counts pass.

    Arguments:
        txt (str): LLVM output.

    Returns:
        dict: key, value pairs, where key is instruction type and value is
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


def instcounts2ratios(counts: dict) -> dict:
    """
    Convert instruction counts to instruction densities.

    If, for example, there are 10 instructions and 2 addition instructions,
    then the instruction density of add operations is .2.

    Arguments:
        counts (dict): Key value pairs of instruction types and counts.

    Returns:
        ratios (dict): Key value pairs of instruction types and densities.
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


def bytecode_features(bc: str, id: str='anon') -> dict:
    """
    Extract features from bytecode.

    Arguments:
        bc (str): LLVM bytecode.
        id (str, optional): Name of OpenCL source.

    Returns:
        dict: Key value pairs of instruction types and densities.

    Raises:
        OptException: If LLVM opt pass errors.
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

    Arguments:
        src (str): OpenCL source.
        id (str, optional): Name of OpenCL source.

    Returns:
        str: Styled source.

    Raises:
        ClangFormatException: If formatting errors.
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


def print_bytecode_features(db_path: str) -> None:
    """
    Print Bytecode features.

    Arguments:
        db_path: Path to dataset.
    """
    db = dbutil.connect(db_path)
    c = db.cursor()

    c.execute('SELECT sha,contents FROM Bytecodes')
    query = c.fetchall()

    uniq_features = set()
    for row in query:
        sha, contents = row

        features = bytecode_features(contents)
        # Add the table key
        features['sha'] = sha
        for key in features.keys():
            uniq_features.add(key)

    log.info('Features:')
    for feature in uniq_features:
        log.info('        ', feature)


def verify_bytecode_features(bc_features: dict, id: str='anon') -> None:
    """
    Verify LLVM bytecode features.

    Arguments:
        bc_features (dict): Bytecode features.
        id (str, optional): Name of OpenCL source.

    Raises:
        InstructionCountException: If verification errors.
    """
    # The minimum number of instructions before a kernel is discarded
    # as ugly.
    min_num_instructions = 0
    try:
        num_instructions = bc_features['instructions_of_all_types']
    except KeyError:
        num_instructions = 0

    if num_instructions < min_num_instructions:
        raise InstructionCountException(
            'Code contains {} instructions. The minimum allowed is {}'
            .format(num_instructions, min_num_instructions))


def ensure_has_code(src: str) -> str:
    """
    Check that file contains actual executable code.

    Arguments:
        src (str): OpenCL source, must be preprocessed.

    Raises:
        NoCodeException: If kernel is empty.
    """
    if len(src.split('\n')) < 3:
        raise NoCodeException

    return src


def sanitize_prototype(src: str) -> str:
    """
    Sanitize OpenCL prototype.

    Ensures that OpenCL prototype fits on a single line.

    Arguments:
        src (str): OpenCL source.

    Returns:
        str: Sanitized OpenCL source.
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


def preprocess(src: str, id: str='anon', use_shim: bool=True) -> str:
    """
    Preprocess an OpenCL source. There are three possible outcomes:

    1. Good. Code is preprocessed and ready to be put into a training set.
    2. Bad. Code can't be preprocessed (i.e. it's "bad" OpenCL).
    3. Ugly. Code can be preprocessed but isn't useful for training
       (e.g. it's an empty file).

    Arguments:
        src (str): The source code as a string.
        id (str, optional): An identifying name for the source code
            (used in exception messages).
        use_shim (bool, optional): Inject shim header.

    Returns:
        str: Preprocessed source code as a string.

    Raises:
        BadCodeException: If code is bad (see above).
        UglyCodeException: If code is ugly (see above).
        clgen.InternalException: In case of some other error.
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

    return src


def _preprocess_db_worker(job: dict) -> None:
    """Database worker thread"""
    db_path = job["db_in"]
    db_index_range = job["db_index_range"]
    outpath = job["json_out"]
    log.debug("worker", os.getpid(), outpath)

    db = dbutil.connect(db_path)
    c = db.cursor()
    split_start, split_end = db_index_range
    split_size = split_end - split_start

    # get the files to preprocess
    c.execute('SELECT id,contents FROM ContentFiles LIMIT {} OFFSET {}'
              .format(split_size, split_start))

    with open(outpath, 'wb') as outfile:
        for row in c.fetchall():
            id, contents = row

            # Get checksum of cached file:
            c.execute('SELECT id FROM PreprocessedFiles WHERE id=?', (id,))
            result = c.fetchone()
            cached_id = result[0] if result else None

            # Check that file is modified:
            if id != cached_id:
                try:
                    # Try and preprocess it:
                    contents = preprocess(contents, id)
                    status = 0
                except BadCodeException as e:
                    contents = str(e)
                    status = 1
                except UglyCodeException as e:
                    contents = str(e)
                    status = 2

                # write result to json
                line = json.dumps([id, status, contents]).encode('utf-8')
                outfile.write(line)
                outfile.write('\n'.encode('utf-8'))

    c.close()
    db.close()


def preprocess_contentfiles(db_path: str, max_num_workers: int=cpu_count(),
                            attempt: int=1) -> None:
    """
    Preprocess OpenCL dataset.

    Arguments:
        db_path (str): OpenCL kernels dataset.
        max_num_workers (int, optional): Number of processes to spawn.
    """
    def _finalize(db_path, cache):
        """Tidy up after worker threads finish"""
        log.debug("worker finalize")

        db = dbutil.connect(db_path)
        c = db.cursor()

        # import results from worker threads
        for outpath in fs.ls(cache.path, abspaths=True):
            with open(outpath) as infile:
                for line in infile:
                    c.execute('INSERT OR REPLACE INTO PreprocessedFiles '
                              'VALUES(?,?,?)', json.loads(line))

        # write changes to database and remove cache
        db.commit()
        db.close()
        cache.empty()

    if attempt >= MAX_OS_RETRIES:
        raise clgen.InternalError("failed to preprocess files")

    num_contentfiles = dbutil.num_rows_in(db_path, 'ContentFiles')
    num_preprocessedfiles = dbutil.num_rows_in(db_path, 'PreprocessedFiles')
    log.info("{n} ({r:.1%}) files need preprocessing".format(
        n=num_contentfiles - num_preprocessedfiles,
        r=(num_contentfiles - num_preprocessedfiles) / num_contentfiles))

    # split into mulitple jobs of a maximum size
    jobsize = min(512, num_contentfiles)
    numjobs = math.ceil(num_contentfiles / jobsize)
    for j, offset in enumerate(range(0, num_contentfiles, jobsize)):
        num_preprocessedfiles = dbutil.num_rows_in(db_path, 'PreprocessedFiles')
        num_workers = min(num_contentfiles, max_num_workers)
        files_per_worker = math.ceil(jobsize / num_workers)

        # temporary cache used for worker thread results
        cache = Cache("{pid}.preprocess".format(pid=os.getpid()))
        # each worker thread receives a range of database indices to preprocess,
        # and a JSON file to write results into
        jobs = [{
            "db_in": db_path,
            "db_index_range": (offset + i * files_per_worker,
                               offset + i * files_per_worker + files_per_worker),
            "json_out": fs.path(cache.path, "{i}.json".format(i=i))
        } for i in range(num_workers)]

        # spool up worker threads then finalize
        log.info('job {j} of {numjobs}: spawning {num_workers} worker threads '
                 'to process {jobsize} files ...'.format(**vars()))
        try:
            with clgen.terminating(Pool(num_workers)) as pool:
                pool.map(_preprocess_db_worker, jobs)
        except OSError as e:
            _finalize(db_path, cache)
            log.error(e)

            # Try again with fewer threads.
            # See: https://github.com/ChrisCummins/clgen/issues/64
            max_num_workers = max(int(max_num_workers / 2), 1)
            preprocess_contentfiles(db_path, max_num_workers=max_num_workers,
                                    attempt=attempt + 1)
        except Exception as e:
            _finalize(db_path, cache)
            raise e
        _finalize(db_path, cache)


def preprocess_file(path: str, inplace: bool=False) -> None:
    """
    Preprocess a file.

    Prints output to stdout by default. If preprocessing fails, this function
    exits.

    Arguments:
        path (str): String path to file.
        inplace (bool, optional): If True, overwrite input file.
    """
    with open(path) as infile:
        contents = infile.read()
    try:
        out = preprocess(contents)
        if inplace:
            with open(path, 'w') as outfile:
                outfile.write(out)
        else:
            print(out)
    except BadCodeException as e:
        log.fatal(e, ret=1)
    except UglyCodeException as e:
        log.fatal(e, ret=2)


def _preprocess_inplace_worker(path: str):
    """worker function for preprocess_inplace()"""
    log.info('preprocess', path)
    preprocess_file(path, inplace=True)


MAX_OS_RETRIES = 10


def preprocess_inplace(paths: str, max_num_workers: int=cpu_count(),
                       attempt: int=1) -> None:
    """
    Preprocess a list of files in place.

    Arguments:
        paths (str[]): List of paths.
        max_num_workers (int, optional): Number of processes to spawn.
    """
    if attempt >= MAX_OS_RETRIES:
        raise clgen.InternalError("Failed to process files")

    num_workers = min(len(paths), max_num_workers)

    try:
        log.info('spawned', num_workers, 'worker threads to process',
                 len(paths), 'files ...')
        with clgen.terminating(Pool(num_workers)) as pool:
            pool.map(_preprocess_inplace_worker, paths)
    except OSError as e:
        log.error(e)

        # Try again with fewer threads.
        # See: https://github.com/ChrisCummins/clgen/issues/64
        max_num_workers = max(int(max_num_workers / 2), 1)
        preprocess_inplace(paths, max_num_workers=max_num_workers,
                           attempt=attempt + 1)



def preprocess_db(db_path: str) -> bool:
    """
    Preprocess database contents.

    Arguments:
        db_path (str): Path to database.

    Returns:
        bool: True if modified, false if no work needed.
    """
    db = dbutil.connect(db_path)

    modified = dbutil.is_modified(db)
    if modified:
        preprocess_contentfiles(db_path)
        dbutil.set_modified_status(db, modified)
        return True
    else:
        return False
