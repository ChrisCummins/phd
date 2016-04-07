#!/usr/bin/env python3
#
# Preprocess the raw dataset.
#
# TODO:
#
# Possible ways to clean source code:
#
# Use UNIX line endings
# Strip empty lines
# Remove comments
# Enforce style (e.g. always use {} on if/else, auto-indentation)
# Rewrite with single-char variable names
#
# Extrapolated data:
#
# Try compiling each source to LLVM bytecode
# For those that build, run static analysis to generate feature vectors
#
import locale
import os
import shutil
import sqlite3
import sys

from subprocess import Popen,PIPE
from multiprocessing import Pool


def usage():
    print('Usage: {} <db>'.format(sys.argv[0]))


# Write OpenCL files.
#
def ocl_writer_worker(db_path):
    print('ocl writer worker ...')

    out_dir = 'cl'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    db = sqlite3.connect(db_path)
    c = db.cursor()

    c.execute('SELECT sha,path,contents FROM OpenCLFiles GROUP BY sha')
    query = c.fetchall()

    files_added_counter = 0
    files_skipped_counter = 0
    files_error_counter = 0
    for row in query:
        sha, path, contents = row
        _, extension = os.path.splitext(path)
        try:
            out_path = out_dir + '/' + sha + extension
            if os.path.exists(out_path):
                files_skipped_counter += 1
            else:
                with open(out_path, 'wb') as out:
                    out.write(contents)
                files_added_counter += 1
        except Exception as e:
            out_path = out_dir + '/' + sha + '.error'
            with open(out_path, 'w') as out:
                out.write(str(e) + '\n')
            files_error_counter += 1
    return (
        'ocl files stats: {} added, {} skipped, {} errors.'
        .format(files_added_counter, files_skipped_counter,
                files_error_counter))


def preprocess_cl(src):
    clang = os.path.expanduser('~/phd/tools/llvm/build/bin/clang')
    libclc = os.path.expanduser('~/phd/extern/libclc')

    cmd = [
        clang, '-Dcl_clang_storage_class_specifiers',
        '-I', '{}/generic/include'.format(libclc),
        '-include', '{}/generic/include/clc/clc.h'.format(libclc),
        '-target', 'nvptx64-nvidia-nvcl',
        '-x', 'cl', '-E',
        '-c', '-', '-o', '-'
    ]

    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate(src)

    if process.returncode != 0:
        raise Exception(stderr)
    return stdout


def compile_cl_bytecode(src):
    clang = os.path.expanduser('~/phd/tools/llvm/build/bin/clang')
    libclc = os.path.expanduser('~/phd/extern/libclc')

    cmd = [
        clang, '-Dcl_clang_storage_class_specifiers',
        '-I', '{}/generic/include'.format(libclc),
        '-include', '{}/generic/include/clc/clc.h'.format(libclc),
        '-target', 'nvptx64-nvidia-nvcl',
        '-x', 'cl', '-emit-llvm', '-S',
        '-c', '-', '-o', '-'
    ]

    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate(src)

    if process.returncode != 0:
        raise Exception(stderr)
    return stdout


# Compile OpenCL files into bytecode.
#
def ocl_builder_worker(db_path):
    print('ocl builder worker ...')

    out_dir = 'bc'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    db = sqlite3.connect(db_path)
    c = db.cursor()

    c.execute('SELECT sha,path,contents FROM OpenCLFiles GROUP BY sha')
    query = c.fetchall()

    counter = 0
    files_added_counter = 0
    files_skipped_counter = 0
    files_error_counter = 0
    for row in query:
        counter += 1
        sha, path, contents = row
        print('\r\033[K', counter, path, end='')
        sys.stdout.flush()

        out_path = out_dir + '/' + sha + '.bc'
        err_path = out_dir + '/' + sha + '.error'

        # Check to see if we've already compiled it.
        if (os.path.exists(out_path) or
            os.path.exists(err_path)):
            files_skipped_counter += 1
        else:
            try:
                bc = compile_cl_bytecode(contents)

                # Add to database.
                c = db.cursor()
                c.execute('INSERT INTO Bytecodes VALUES(?,?)',
                          (sha,bc))
                db.commit()

                # Write file.
                with open(out_path, 'wb') as out:
                    out.write(bc)
                files_added_counter += 1
            except Exception as e:
                # Add to database.
                c = db.cursor()
                c.execute('INSERT INTO BytecodeErrors VALUES(?,?)',
                          (sha, str(e)))
                db.commit()

                out_path = out_dir + '/' + sha + '.error'
                with open(out_path, 'w') as out:
                    out.write(str(e) + '\n')
                files_error_counter += 1

    # Clear output
    print('\r\033[K', end='')
    sys.stdout.flush()
    return (
        'ocl bytecode stats: {} added, {} skipped, {} errors.'
        .format(files_added_counter, files_skipped_counter,
                files_error_counter))


# Preprocess OpenCL files.
#
def ocl_preprocessor_worker(db_path):
    print('ocl preprocessor worker ...')

    db = sqlite3.connect(db_path)
    c = db.cursor()

    c.execute('SELECT sha,path,contents FROM OpenCLFiles GROUP BY sha')
    query = c.fetchall()

    files_added_counter = 0
    files_skipped_counter = 0
    files_error_counter = 0
    for row in query:
        sha, path, contents = row

        # Check to see if we've already compiled it.
        c = db.cursor()
        c.execute('SELECT sha FROM Preprocessed WHERE sha=?', (sha,))
        is_preprocessed = c.fetchone()
        c.execute('SELECT sha FROM PreprocessedErrors WHERE sha=?', (sha,))
        is_preprocessed_error = c.fetchone()

        if (is_preprocessed or is_preprocessed_error):
            files_skipped_counter += 1
        else:
            try:
                cl = preprocess_cl(contents)

                # Add to database.
                c.execute('INSERT INTO Preprocessed VALUES(?,?)',
                          (sha,cl))
                files_added_counter += 1
            except Exception as e:
                # Add to database.
                c = db.cursor()
                c.execute('INSERT INTO PreprocessedErrors VALUES(?,?)',
                          (sha, str(e)))
                files_error_counter += 1
            db.commit()

    return (
        'ocl preprocessor stats: {} added, {} skipped, {} errors.'
        .format(files_added_counter, files_skipped_counter,
                files_error_counter))


def main():
    locale.setlocale(locale.LC_ALL, 'en_GB')

    if len(sys.argv) != 2:
        usage()
        sys.exit(1)

    db_path = sys.argv[1]

    db_path = sys.argv[1]

    # Worker process pool
    pool, jobs = Pool(processes=4), []
    jobs.append(pool.apply_async(ocl_writer_worker, (db_path,)))
    jobs.append(pool.apply_async(ocl_preprocessor_worker, (db_path,)))
    jobs.append(pool.apply_async(ocl_builder_worker, (db_path,)))

    # Wait for jobs to finish
    [job.wait() for job in jobs]

    # Print job output
    print()
    for job in jobs:
        output = job.get()
        print(output)


if __name__ == '__main__':
    main()
