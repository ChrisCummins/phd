#!/usr/bin/env python3
#
# Exploratory analysis of preprocessed dataset.
#
# TODO: Graphs
#
#   Number of OpenCL repos over time
#   Distribution of stars and OpenCL file counts
#   Distribution of repo star counts
#   Distribution of file star counts
#   Distribution of repo forks
#   Distribution of times since last changed
#   Distribution of file sizes
#   Distribution of files per repo
#
import locale
import os
import shutil
import sqlite3
import sys

from multiprocessing import Pool

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


def usage():
    print('Usage: {} <db>'.format(sys.argv[0]))


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


# Generate dataset stats.
#
def stats_worker(db_path):
    print("stats worker ...")
    db = sqlite3.connect(sys.argv[1])
    c = db.cursor()
    stats = []

    c.execute("SELECT Count(*) from Repositories")
    nb_repos = c.fetchone()[0]
    stats.append(('Number of repositories visited', bigint(nb_repos)))
    stats.append(('',''))

    c.execute("SELECT Count(DISTINCT repo_url) from OpenCLFiles")
    nb_ocl_repos = c.fetchone()[0]
    stats.append(('Number of OpenCL repositories', bigint(nb_ocl_repos)))

    c.execute("SELECT Count(*) from OpenCLFiles")
    nb_ocl_files = c.fetchone()[0]
    stats.append(('Number of OpenCL files', bigint(nb_ocl_files)))

    c.execute("SELECT Count(*) from Repositories WHERE fork=1")
    nb_forks = c.fetchone()[0]
    ratio_forks = nb_forks / nb_repos
    stats.append(('Number of forked repositories',
                  bigint(nb_forks) +
                  ' ({:.0f}%)'.format(ratio_forks * 100)))
    stats.append(('',''))

    c.execute("SELECT Count(DISTINCT sha) from OpenCLFiles")
    nb_uniq_ocl_files = c.fetchone()[0]
    ratio_uniq_ocl_files = nb_uniq_ocl_files / nb_ocl_files
    stats.append(('Number of unique OpenCL files',
                  bigint(nb_uniq_ocl_files) +
                  ' ({:.0f}%)'.format(ratio_uniq_ocl_files * 100)))

    c.execute("SELECT contents FROM OpenCLFiles")
    code = c.fetchall()
    code_lcs = [len(decode(x[0]).split('\n')) for x in code]
    code_lcs.sort()
    stats.append(('Total OpenCL line count', bigint(sum(code_lcs))))

    avg_nb_ocl_files_per_repo = nb_ocl_files / nb_ocl_repos
    stats.append(('OpenCL files per repository',
                  'avg: {:.2f}'.format(avg_nb_ocl_files_per_repo)))

    avg_code_lcs = sum(code_lcs) / len(code_lcs)
    midpoint = int(len(code_lcs) / 2)
    if len(code_lcs) % 2 == 1:
        med_code_lcs = code_lcs[midpoint]
    else:
        med_code_lcs = (code_lcs[midpoint - 1] + code_lcs[midpoint]) / 2.0
    stats.append(('OpenCL line counts',
                  'min: {}, med: {}, avg: {}, max: {}'.format(
                      bigint(code_lcs[0]), bigint(med_code_lcs),
                      bigint(avg_code_lcs),
                      bigint(code_lcs[len(code_lcs) - 1]))))

    stats.append(('',''))

    c.execute("SELECT Count(*) from Bytecodes")
    nb_bc_files = c.fetchone()[0]
    ratio_bc_files = nb_bc_files / nb_uniq_ocl_files
    stats.append(('Number of Bytecode files',
                  bigint(nb_bc_files) +
                  ' ({:.0f}%)'.format(ratio_bc_files * 100)))


    return stats


def main():
    locale.setlocale(locale.LC_ALL, 'en_GB')

    if len(sys.argv) != 2:
        usage()
        sys.exit(1)

    db_path = sys.argv[1]

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
