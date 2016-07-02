#
# Preprocess the raw dataset.
#
# TODO:
#
#   More strict formatting, which enforces things like one blank line
#   between each top level {} block, one parameter per line in
#   function declaration, etc.
#
# Extrapolated data:
#
# Try compiling each source to LLVM bytecode
# For those that build, run static analysis to generate feature vectors
#
import math
import os
import re
import shutil
import sqlite3
import sys

from argparse import ArgumentParser
from functools import partial
from hashlib import md5
from multiprocessing import cpu_count,Pool
from subprocess import Popen,PIPE,STDOUT
from tempfile import NamedTemporaryFile


#
# Custom exceptions:
#
class RewriterException(Exception): pass

# LLVM exceptions:
class LlvmException(Exception): pass
class ClangException(LlvmException): pass
class OptException(LlvmException): pass

# Good, bad, ugly exceptions:
class BadCodeException(Exception): pass
class CodeCompilationException(BadCodeException): pass
class CodeAnalysisException(BadCodeException): pass

class UglyCodeException(Exception): pass
class InstructionCountException(BadCodeException): pass

clang_defines = [
    '__CL_VERSION_1_1__'
]
clang_define_vals = {
    '__OPENCL_VERSION__': 1,
    'BLK_X': 8,
    'BLK_Y': 8,
    'BLOCK_DIM': 2,
    'BLOCK_SIZE': 64,
    'BLOCK_X': 8,
    'BLOCK_Y': 8,
    'BUCKETS': 8,
    'CONCURRENT_THREADS': 128,
    'CUTOFF_VAL': 0.5,
    'DATA_TYPE': 'float',
    'DATATYPE': 'float',
    'ELEMENTS': 1024,
    'ELEMENTS': 16,
    'FLOAT_T': 'float',
    'FLOAT_TYPE': 'float',
    'FORCE_WORK_GROUP_SIZE': 32,
    'GAUSS_RADIUS': 5,
    'GLOBALSIZE_LOG2': 10,
    'HEIGHT': 128,
    'INPUT_WIDTH': 256,
    'ITERATIONS': 1000,
    'LOCAL_MEM_SIZE': 2048,
    'LOCAL_MEMORY_BANKS': 16,
    'LOCAL_SIZE': 128,
    'LOCAL_SIZE_LIMIT': 1024,
    'LOCAL_W': 128,
    'LOCALSIZE_LOG2': 5,
    'LOG2_WARP_SIZE': 5,
    'M_PI': 3.14,
    'Pixel': 'float3',
    'SCREENHEIGHT': 1920,
    'SCREENWIDTH': 1080,
    'SIMD_WIDTH': 32,
    'static': '',
    'THRESHOLD': 0.5,
    'TILE_COLS': 16,
    'TILE_COLS': 16,
    'TILE_DIM': 16,
    'TILE_HEIGHT': 16,
    'TILE_M': 16,
    'TILE_N': 16,
    'TILE_ROWS': 16,
    'TILE_SIZE': 16,
    'TILE_TB_HEIGHT': 16,
    'TILE_WIDTH': 16,
    'TILEH': 16,
    'TILESW': 16,
    'TILEW': 16,
    'TREE_DEPTH': 3,
    'TYPE': 'float',
    'WARPS_PER_GROUP': 8,
    'WORK_GROUP_SIZE': 256,
    'WORK_ITEMS': 128,
    'WORKGROUP_SIZE': 256,
    'WORKGROUPSIZE': 256,
    'WORKSIZE': 128,
    'WORKSIZE': 256,
    'WSIZE': 128,
    'zero': 0,
    'zeroVal': 0,
}
clang_define_args = ['-D{}'.format(d) for d in clang_defines] + [
    '-D{}={}'.format(k,v) for k,v in clang_define_vals.items()
]


def num_rows_in(db, table):
    c = db.cursor()
    c.execute('SELECT Count(*) FROM ' + str(table))
    num_rows = c.fetchone()[0]
    c.close()
    return num_rows


def preprocess_cl(src):
    clang = os.path.expanduser('~/phd/tools/llvm/build/bin/clang')
    libclc = os.path.expanduser('~/phd/extern/libclc')

    cmd = [
        clang, '-Dcl_clang_storage_class_specifiers',
        '-I', '{}/generic/include'.format(libclc),
        '-include', '{}/generic/include/clc/clc.h'.format(libclc),
        '-target', 'nvptx64-nvidia-nvcl'
    ] + clang_define_args + [
        '-x', 'cl', '-E',
        '-c', '-', '-o', '-'
    ]

    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate(src)

    if process.returncode != 0:
        raise ClangException(stderr.decode('utf-8'))

    src = stdout.decode('utf-8')
    lines = src.split('\n')

    # Strip all the includes:
    for i,line in enumerate(lines):
        if line == '# 1 "<stdin>" 2':
            break
    src = '\n'.join(lines[i+1:]).strip()

    # Strip lines beginning with '#' (that's preprocessor
    # stuff):
    src = '\n'.join([line for line in src.split('\n')
                     if not line.startswith('#')])

    return src


def rewrite_cl(in_path):
    ld_path = os.path.expanduser('~/phd/tools/llvm/build/lib/')
    libclc = os.path.expanduser('~/phd/extern/libclc')
    rewriter = os.path.expanduser('~/phd/lab/ml/rewriter')

    extra_args = [
        '-Dcl_clang_storage_class_specifiers',
        '-I{}/generic/include'.format(libclc),
        '-include', '{}/generic/include/clc/clc.h'.format(libclc),
        '-target', 'nvptx64-nvidia-nvcl',
        '-DM_PI=3.14',
        '-xcl'
    ]

    cmd = ([rewriter, in_path ]
           + ['-extra-arg=' + x for x in extra_args] + ['--'])

    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE,
                    env = {'LD_LIBRARY_PATH': ld_path})
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise RewriterException(stderr.decode('utf-8'))

    formatted = clangformat_ocl(stdout)

    return formatted.decode('utf-8')


def compile_cl_bytecode(src):
    clang = os.path.expanduser('~/phd/tools/llvm/build/bin/clang')
    libclc = os.path.expanduser('~/phd/extern/libclc')

    cmd = [
        clang, '-Dcl_clang_storage_class_specifiers',
        '-I', '{}/generic/include'.format(libclc),
        '-include', '{}/generic/include/clc/clc.h'.format(libclc),
        '-target', 'nvptx64-nvidia-nvcl'
    ] + clang_define_args + [
        '-x', 'cl', '-emit-llvm', '-S',
        '-c', '-', '-o', '-'
    ]

    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate(src)

    if process.returncode != 0:
        raise ClangException(stderr.decode('utf-8'))
    return stdout


_instcount_re = re.compile("^(?P<count>\d+) instcount - Number of (?P<type>.+)")


def parse_instcounts(txt):
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


_sql_rm_chars = re.compile('[\(\)]')
_sql_sub_chars = re.compile('-')


def escape_sql_key(key):
    return re.sub(_sql_sub_chars, '_',
                  re.sub(_sql_rm_chars, '', '_'.join(key.split(' '))))


def instcounts2ratios(counts):
    if not len(counts):
        return {}

    ratios = {}
    total_key = "instructions (of all types)"
    non_ratio_keys = [
        total_key
    ]
    total = float(counts[total_key])

    for key in non_ratio_keys:
        ratios[escape_sql_key(key)] = counts[key]

    for key in counts:
        if key not in non_ratio_keys:
            # Copy count
            ratios[escape_sql_key(key)] = counts[key]
            # Insert ratio
            ratios[escape_sql_key('ratio_' + key)] = float(counts[key]) / total

    return ratios


def sql_insert_dict(c, table, data):
    cmd = ("INSERT INTO {table}({cols}) VALUES({vals})"
           .format(table=table,
                   cols=','.join(data.keys()),
                   vals=','.join(['?'] * len(data))))

    vals = tuple(data.values())
    c.execute(cmd, tuple(data.values()))


def bytecode_features(bc):
    opt = os.path.expanduser('~/phd/tools/llvm/build/bin/opt')

    cmd = [
        opt, '-analyze', '-stats', '-instcount', '-'
    ]

    # LLVM pass output pritns to stderr, so we'll pipe stderr to
    # stdout.
    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    stdout, _ = process.communicate(bc)

    if process.returncode != 0:
        raise OptException(stdout.decode('utf-8'))

    instcounts = parse_instcounts(stdout.decode('utf-8'))
    instratios = instcounts2ratios(instcounts)

    return instratios

def clangformat_ocl(src):
    clangformat = os.path.expanduser('~/phd/tools/llvm/build/bin/clang-format')
    cmd = [
        clangformat, '-style=google'
    ]

    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate(src)

    if process.returncode != 0:
        raise Exception(stderr.decode('utf-8'))

    return stdout


def print_bytecode_features(db_path):
    db = sqlite3.connect(db_path)
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

    print('Features:')
    for feature in uniq_features:
        print('        ', feature)


# 3 possible outcomes:
#
#   1. Good. Code is preprocessed and ready to be put into a training set.
#   2. Bad. Code can't be preprocessed.
#   3. Ugly. Code can be preprocessed, but isn't useful for training.
#
def preprocess(src):
    srcbuf = src.encode('utf-8')

    # Check that code compiles:
    try:
        bc = compile_cl_bytecode(srcbuf)
    except ClangException as e:
        raise CodeCompilationException(e)

    # Check that feature extraction works:
    try:
        bc_features = bytecode_features(bc)
    except OptException as e:
        raise CodeAnalysisException(e)

    # Check that code contains more than a minimum number of instructions:
    try:
        num_instructions = bc_features['instructions_of_all_types']
    except KeyError:
        num_instructions = 0

    min_num_instructions = 2
    if num_instructions < min_num_instructions:
        raise InstructionCountException(
            'Code contains {} instructions. The minimum allowed is {})'
            .format(num_instructions, min_num_instructions))

    # Run source through preprocesor:
    try:
        src = preprocess_cl(srcbuf)
    except ClangException as e:
        raise CodeCompilationException(e)

    # Rewrite source:
    with NamedTemporaryFile('w', suffix='.cl') as tmp:
        # Write to file:
        tmp.write(src)
        tmp.flush()

        # Perform rewrite:
        try:
            src = rewrite_cl(tmp.name)
        except RewriterException as e:
            raise CodeCompilationException(e)

    return src

def md5sum(t):
    return md5(t).hexdigest()

class md5sum_aggregator:
    def __init__(self):
        self.md5 = md5()

    def step(self, value):
        self.md5.update(str(value).encode('utf-8'))

    def finalize(self):
        return self.md5.hexdigest()


class linecount_aggregator:
    def __init__(self):
        self.count = 0

    def step(self, value):
        self.count += len(value.split('\n'))

    def finalize(self):
        return self.count


class charcount_aggregator:
    def __init__(self):
        self.count = 0

    def step(self, value):
        self.count += len(value)

    def finalize(self):
        return self.count


def is_modified(db):
    c = db.cursor()

    c.execute("SELECT value FROM Meta WHERE key='preprocessed_checksum'")
    result = c.fetchone()
    cached_checksum = result[0] if result else None

    c.execute('SELECT MD5SUM(id) FROM ContentFiles')
    checksum = c.fetchone()[0]
    c.close()

    return False if cached_checksum == checksum else checksum


def set_modified_status(db, checksum):
    c = db.cursor()
    c.execute("INSERT OR REPLACE INTO Meta VALUES (?,?)",
              ('preprocessed_checksum', checksum))
    db.commit()
    c.close()


def preprocess_split(db_path, split):
    db = sqlite3.connect(db_path)
    c = db.cursor()
    split_start, split_end = split
    split_size = split_end - split_start

    c.execute('SELECT id,contents FROM ContentFiles LIMIT {} OFFSET {}'
              .format(split_size, split_start))
    rows = c.fetchall()
    c.close()

    for row in rows:
        id, contents = row
        c = db.cursor()

        # Get checksum of cached file:
        c.execute('SELECT id FROM PreprocessedFiles WHERE id=?', (id,))
        result = c.fetchone()
        cached_id = result[0] if result else None

        # Check that file is modified:
        if id != cached_id:
            try:
                # Try and preprocess it:
                contents = preprocess(contents)
                status = 0
            except BadCodeException as e:
                contents = str(e)
                status = 1
            except UglyCodeException as e:
                contents = str(e)
                status = 2
            c.execute('INSERT OR REPLACE INTO PreprocessedFiles '
                      'VALUES(?,?,?)',
                      (id,status,contents))
            db.commit()
        c.close()


def preprocess_contentfiles(db_path):
    db = sqlite3.connect(db_path)
    num_contentfiles = num_rows_in(db, 'ContentFiles')
    num_preprocessedfiles = num_rows_in(db, 'PreprocessedFiles')
    db.close()

    num_workers = round(cpu_count() * 1.5)

    files_per_worker = math.ceil(num_contentfiles / num_workers)

    splits = [(i * files_per_worker,
               i * files_per_worker + files_per_worker)
              for i in range(num_workers)]

    with Pool(num_workers) as pool:
        print('spawning', num_workers, 'worker threads to process',
              num_contentfiles - num_preprocessedfiles, 'files ...')
        worker = partial(preprocess_split, db_path)
        pool.map(worker, splits)


def preprocess_and_print(contents):
    try:
        print(preprocess(contents))
    except BadCodeException as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except UglyCodeException as e:
        print(e, file=sys.stderr)
        sys.exit(2)


def preprocess_file(path):
    with open(input_path) as infile:
        preprocess_and_print(infile.read())


def preprocess_db(db_path):
    db = sqlite3.connect(db_path)
    db.create_aggregate("MD5SUM", 1, md5sum_aggregator)
    db.create_aggregate("LC", 1, linecount_aggregator)
    db.create_aggregate("CC", 1, charcount_aggregator)

    modified = is_modified(db)
    if modified:
        preprocess_contentfiles(db_path)
        set_modified_status(db, modified)
        print('done.')
    else:
        print('nothing to be done.')
