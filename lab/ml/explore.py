#!/usr/bin/env python3
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

img_dir = 'img'


def decode(code):
    try:
        return code.decode('utf-8')
    except UnicodeDecodeError:
        return ''


def median(sorted_arr):
    n = len(array)

    midpoint = int(n / 2)
    if n % 2 == 1:
        return sorted_vals[midpoint]
    else:
        return (sorted_vals[midpoint - 1] + sorted_vals[midpoint]) / 2.0


def bigint(n):
    return locale.format('%d', round(n), grouping=True)


def seq_stats(sorted_arr):
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
    ratio_uniq_ocl_files = nb_uniq_ocl_files / nb_ocl_files
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
    ratio_pp_files = nb_pp_files / nb_ocl_files
    stats.append(('Number of good preprocessed files',
                  bigint(nb_pp_files) +
                  ' ({:.0f}%)'.format(ratio_pp_files * 100)))

    c.execute('SELECT contents FROM PreprocessedFiles WHERE status=0')
    bc = c.fetchall()
    pp_lcs = [len(x[0].split('\n')) for x in bc]
    pp_lcs.sort()
    pp_lc = sum(pp_lcs)
    ratio_pp_lcs = pp_lc / code_lc
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
    plt.title('Distribution of OpenCL file lengths')
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
    data = [x for x in stars if x < 100]

    sns.distplot(data, bins=20, kde=False)
    plt.xlabel('GitHub Stargazer count')
    plt.ylabel('Number of files')
    plt.title('Stargazers per file')
    plt.savefig(out_path)


def main():
    locale.setlocale(locale.LC_ALL, 'en_GB.utf-8')

    parser = ArgumentParser()
    parser.add_argument('input', help='path to SQL input dataset')
    args = parser.parse_args()

    db_path = args.input

    db = sqlite3.connect(db_path)

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # Worker process pool
    pool, jobs = Pool(processes=4), []
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

if __name__ == '__main__':
    main()
