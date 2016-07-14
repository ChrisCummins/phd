#
# Exploratory analysis of preprocessed dataset.
#
# TODO:
#
#   Graphs:
#
#     Distribution of file sizes
#
import locale
import os
import shutil
import sqlite3
import sys

from argparse import ArgumentParser
from multiprocessing import Pool

import smith
from smith import dbutil


img_dir = 'img'


def decode(code):
    try:
        return code.decode('utf-8')
    except UnicodeDecodeError:
        return ''


def div(x, y):
    """
    Zero-safe division.

    :param x: Numerator
    :param y: Denominator
    :return: x / y
    """
    try:
        return x / y
    except ZeroDivisionError:
        return 0


def median(sorted_arr):
    n = len(array)
    if not n:
        return 0

    midpoint = int(n / 2)
    if n % 2 == 1:
        return sorted_vals[midpoint]
    else:
        return (sorted_vals[midpoint - 1] + sorted_vals[midpoint]) / 2.0


def bigint(n):
    return locale.format('%d', round(n), grouping=True)


def seq_stats(sorted_arr):
    sorted_arr = sorted_arr or [0]
    avg = sum(sorted_arr) / len(sorted_arr)
    midpoint = int(len(sorted_arr) / 2)
    if len(sorted_arr) % 2 == 1:
        median = sorted_arr[midpoint]
    else:
        median = (sorted_arr[midpoint - 1] + sorted_arr[midpoint]) / 2.0
    return (
        'min: {}, med: {}, avg: {}, max: {}'.format(
            bigint(sorted_arr[0]), bigint(median), bigint(avg),
            bigint(sorted_arr[len(sorted_arr) - 1])))


# Generate dataset stats.
#
def stats_worker(db_path):
    print("stats worker ...")
    db = sqlite3.connect(db_path)
    c = db.cursor()
    stats = []

    c.execute("SELECT Count(*) from ContentFiles")
    nb_ocl_files = c.fetchone()[0]
    stats.append(('Number of content files', bigint(nb_ocl_files)))
    stats.append(('',''))

    # ContentFiles
    c.execute("SELECT Count(DISTINCT id) from ContentFiles")
    nb_uniq_ocl_files = c.fetchone()[0]
    ratio_uniq_ocl_files = div(nb_uniq_ocl_files, nb_ocl_files)
    stats.append(('Number of unique content files',
                  bigint(nb_uniq_ocl_files) +
                  ' ({:.0f}%)'.format(ratio_uniq_ocl_files * 100)))

    c.execute("SELECT contents FROM ContentFiles")
    code = c.fetchall()
    code_lcs = [len(x[0].split('\n')) for x in code]
    code_lcs.sort()
    code_lc = sum(code_lcs)
    stats.append(('Total content line count', bigint(code_lc)))

    stats.append(('Content file line counts', seq_stats(code_lcs)))
    stats.append(('',''))

    # Preprocessed
    c.execute("SELECT Count(*) FROM PreprocessedFiles WHERE status=0")
    nb_pp_files = c.fetchone()[0]
    ratio_pp_files = div(nb_pp_files, nb_ocl_files)
    stats.append(('Number of good preprocessed files',
                  bigint(nb_pp_files) +
                  ' ({:.0f}%)'.format(ratio_pp_files * 100)))

    c.execute('SELECT contents FROM PreprocessedFiles WHERE status=0')
    bc = c.fetchall()
    pp_lcs = [len(x[0].split('\n')) for x in bc]
    pp_lcs.sort()
    pp_lc = sum(pp_lcs)
    ratio_pp_lcs = div(pp_lc, code_lc)
    stats.append(('Lines of good preprocessed code',
                  bigint(pp_lc) +
                  ' ({:.0f}%)'.format(ratio_pp_lcs * 100)))

    stats.append(('Good preprocessed line counts',
                  seq_stats(pp_lcs)))
    stats.append(('',''))

    return stats


# Generate dataset stats.
#
def gh_stats_worker(db_path):
    print("stats worker ...")
    db = sqlite3.connect(db_path)
    c = db.cursor()
    stats = []

    # Repositories
    c.execute("SELECT Count(*) from Repositories")
    nb_repos = c.fetchone()[0]
    stats.append(('Number of repositories visited', bigint(nb_repos)))
    stats.append(('',''))

    c.execute("SELECT Count(DISTINCT repo_url) from ContentMeta")
    nb_ocl_repos = c.fetchone()[0]
    stats.append(('Number of content file repositories', bigint(nb_ocl_repos)))

    c.execute('SELECT Count(*) FROM Repositories WHERE fork=1 AND url IN '
              '(SELECT repo_url FROM ContentMeta)')
    nb_forks = c.fetchone()[0]
    ratio_forks = div(nb_forks, nb_ocl_repos)
    stats.append(('Number of forked repositories',
                  bigint(nb_forks) +
                  ' ({:.0f}%)'.format(ratio_forks * 100)))

    c.execute("SELECT Count(*) from ContentFiles")
    nb_ocl_files = c.fetchone()[0]
    stats.append(('Number of content files', bigint(nb_ocl_files)))
    stats.append(('',''))

    # ContentFiles
    c.execute("SELECT Count(DISTINCT id) from ContentFiles")
    nb_uniq_ocl_files = c.fetchone()[0]
    ratio_uniq_ocl_files = div(nb_uniq_ocl_files, nb_ocl_files)
    stats.append(('Number of unique content files',
                  bigint(nb_uniq_ocl_files) +
                  ' ({:.0f}%)'.format(ratio_uniq_ocl_files * 100)))

    avg_nb_ocl_files_per_repo = div(nb_ocl_files, nb_ocl_repos)
    stats.append(('Content files per repository',
                  'avg: {:.2f}'.format(avg_nb_ocl_files_per_repo)))

    c.execute("SELECT contents FROM ContentFiles")
    code = c.fetchall()
    code_lcs = [len(x[0].split('\n')) for x in code]
    code_lcs.sort()
    code_lc = sum(code_lcs)
    stats.append(('Total content line count', bigint(code_lc)))

    stats.append(('Content file line counts', seq_stats(code_lcs)))
    stats.append(('',''))

    # Preprocessed
    c.execute("SELECT Count(*) FROM PreprocessedFiles WHERE status=0")
    nb_pp_files = c.fetchone()[0]
    ratio_pp_files = div(nb_pp_files, nb_ocl_files)
    stats.append(('Number of good preprocessed files',
                  bigint(nb_pp_files) +
                  ' ({:.0f}%)'.format(ratio_pp_files * 100)))

    c.execute('SELECT contents FROM PreprocessedFiles WHERE status=0')
    bc = c.fetchall()
    pp_lcs = [len(x[0].split('\n')) for x in bc]
    pp_lcs.sort()
    pp_lc = sum(pp_lcs)
    ratio_pp_lcs = div(pp_lc, code_lc)
    stats.append(('Lines of good preprocessed code',
                  bigint(pp_lc) +
                  ' ({:.0f}%)'.format(ratio_pp_lcs * 100)))

    stats.append(('Good preprocessed line counts',
                  seq_stats(pp_lcs)))
    stats.append(('',''))

    return stats


# Plot distribution of OpenCL file line counts.
#
def graph_ocl_lc(db_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(color_codes=True)

    out_path = img_dir + '/ocl_lcs.png'
    print('graph', out_path, '...')
    db = sqlite3.connect(db_path)
    c = db.cursor()

    c.execute("SELECT contents FROM ContentFiles")
    ocl = c.fetchall()
    ocl_lcs = [len(x[0].split('\n')) for x in ocl]

    # Filter range
    data = [x for x in ocl_lcs if x < 500]

    sns.distplot(data, bins=20, kde=False)
    plt.xlabel('Line count')
    plt.ylabel('Number of OpenCL files')
    plt.title('Distribution of source code lengths')
    plt.savefig(out_path)


# Plot distribution of Bytecode line counts.
#
def graph_bc_lc(db_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(color_codes=True)

    out_path = img_dir + '/bc_lcs.png'
    print('graph', out_path, '...')
    db = sqlite3.connect(db_path)
    c = db.cursor()

    c.execute("SELECT contents FROM Bytecodes")
    ocl = c.fetchall()
    ocl_lcs = [len(decode(x[0]).split('\n')) for x in ocl]

    # Filter range
    data = [x for x in ocl_lcs if x < 500]

    sns.distplot(data, bins=20, kde=False)
    plt.xlabel('Line count')
    plt.ylabel('Number of Bytecode files')
    plt.title('Distribution of Bytecode lengths')
    plt.savefig(out_path)


# Plot distribution of stargazers per file.
#
def graph_ocl_stars(db_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(color_codes=True)

    out_path = img_dir + '/ocl_stars.png'
    print('graph', out_path, '...')
    db = sqlite3.connect(db_path)
    c = db.cursor()

    c.execute('SELECT stars FROM ContentMeta LEFT JOIN Repositories '
              'ON ContentMeta.repo_url=Repositories.url')
    stars = [x[0] for x in c.fetchall()]

    # Filter range
    data = [x for x in stars if x < 50]

    sns.distplot(data, bins=20, kde=False)
    plt.xlabel('GitHub Stargazer count')
    plt.ylabel('Number of files')
    plt.title('Stargazers per file')
    plt.savefig(out_path)


def explore(db_path, graph=False):
    locale.setlocale(locale.LC_ALL, 'en_GB.utf-8')

    db = sqlite3.connect(db_path)

    if dbutil.is_github(db):
        db.close()
        explore_gh(db_path)
        return

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # Worker process pool
    pool, jobs = Pool(processes=4), []
    if graph:
        jobs.append(pool.apply_async(graph_ocl_lc, (db_path,)))
        # TODO: If GH dataset:
        # jobs.append(pool.apply_async(graph_ocl_stars, (db_path,)))
    future_stats = pool.apply_async(stats_worker, (db_path,))

    # Wait for jobs to finish
    [job.wait() for job in jobs]

    # Print stats
    print()
    stats = future_stats.get()
    maxlen = max([len(x[0]) for x in stats])
    for stat in stats:
        k,v = stat
        if k:
            print(k, ':', ' ' * (maxlen - len(k) + 2), v, sep='')
        elif v == '':
            print(k)
        else:
            print()


def explore_gh(db_path, graph=False):
    locale.setlocale(locale.LC_ALL, 'en_GB.utf-8')

    db = sqlite3.connect(db_path)

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # Worker process pool
    pool, jobs = Pool(processes=4), []
    if graph:
        jobs.append(pool.apply_async(graph_ocl_lc, (db_path,)))
        jobs.append(pool.apply_async(graph_ocl_stars, (db_path,)))

    future_stats = pool.apply_async(gh_stats_worker, (db_path,))

    # Wait for jobs to finish
    [job.wait() for job in jobs]

    # Print stats
    print()
    stats = future_stats.get()
    maxlen = max([len(x[0]) for x in stats])
    for stat in stats:
        k,v = stat
        if k:
            print(k, ':', ' ' * (maxlen - len(k) + 2), v, sep='')
        elif v == '':
            print(k)
        else:
            print()
