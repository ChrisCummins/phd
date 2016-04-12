#!/usr/bin/env python3
#
# Exploratory analysis of preprocessed dataset.
#
# TODO: Graphs
#
#   Number of OpenCL repos over time
#   Distribution of stars and OpenCL file counts
#   Distribution of repo star counts
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

img_dir = 'img'


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

    # Repositories
    c.execute("SELECT Count(*) from Repositories")
    nb_repos = c.fetchone()[0]
    stats.append(('Number of repositories visited', bigint(nb_repos)))
    stats.append(('',''))

    c.execute("SELECT Count(DISTINCT repo_url) from ContentFiles")
    nb_ocl_repos = c.fetchone()[0]
    stats.append(('Number of content file repositories', bigint(nb_ocl_repos)))

    c.execute('SELECT Count(*) FROM Repositories WHERE fork=1 AND url IN '
              '(SELECT repo_url FROM ContentFiles)')
    nb_forks = c.fetchone()[0]
    ratio_forks = nb_forks / nb_ocl_repos
    stats.append(('Number of forked repositories',
                  bigint(nb_forks) +
                  ' ({:.0f}%)'.format(ratio_forks * 100)))

    c.execute("SELECT Count(*) from ContentFiles")
    nb_ocl_files = c.fetchone()[0]
    stats.append(('Number of content files', bigint(nb_ocl_files)))
    stats.append(('',''))

    # ContentFiles
    c.execute("SELECT Count(DISTINCT sha) from ContentFiles")
    nb_uniq_ocl_files = c.fetchone()[0]
    ratio_uniq_ocl_files = nb_uniq_ocl_files / nb_ocl_files
    stats.append(('Number of unique content files',
                  bigint(nb_uniq_ocl_files) +
                  ' ({:.0f}%)'.format(ratio_uniq_ocl_files * 100)))

    avg_nb_ocl_files_per_repo = nb_ocl_files / nb_ocl_repos
    stats.append(('Content files per repository',
                  'avg: {:.2f}'.format(avg_nb_ocl_files_per_repo)))

    c.execute("SELECT contents FROM ContentFiles")
    code = c.fetchall()
    code_lcs = [len(x[0].split('\n')) for x in code]
    code_lcs.sort()
    stats.append(('Total content line count', bigint(sum(code_lcs))))

    stats.append(('Content file line counts', seq_stats(code_lcs)))
    stats.append(('',''))

    # Preprocessed
    c.execute("SELECT Count(*) from Preprocessed")
    nb_pp_files = c.fetchone()[0]
    ratio_pp_files = nb_pp_files / nb_uniq_ocl_files
    stats.append(('Number of Preprocessed files',
                  bigint(nb_pp_files) +
                  ' ({:.0f}%)'.format(ratio_pp_files * 100)))

    c.execute('SELECT contents FROM Preprocessed')
    bc = c.fetchall()
    pp_lcs = [len(decode(x[0]).split('\n')) for x in bc]
    pp_lcs.sort()
    stats.append(('Total line count of Preprocessed', bigint(sum(pp_lcs))))

    stats.append(('Preprocessed line counts',
                  seq_stats(pp_lcs)))
    stats.append(('',''))

    # Bytecodes
    c.execute('SELECT Count(*) from Bytecodes')
    nb_bc_files = c.fetchone()[0]
    ratio_bc_files = nb_bc_files / nb_uniq_ocl_files
    stats.append(('Number of Bytecode files',
                  bigint(nb_bc_files) +
                  ' ({:.0f}%)'.format(ratio_bc_files * 100)))

    c.execute('SELECT ContentFiles.contents FROM Bytecodes '
              'LEFT JOIN ContentFiles ON Bytecodes.sha=ContentFiles.sha '
              'GROUP BY ContentFiles.sha')
    bc_ocl = c.fetchall()
    bc_ocl_lcs = [len(x[0].split('\n')) for x in bc_ocl]
    bc_ocl_lcs.sort()
    stats.append(('Total line count of Bytecode OpenCL sources',
                  bigint(sum(bc_ocl_lcs))))

    c.execute('SELECT contents FROM Bytecodes')
    bc = c.fetchall()
    bc_lcs = [len(decode(x[0]).split('\n')) for x in bc]
    bc_lcs.sort()
    stats.append(('Total line count of Bytecodes', bigint(sum(bc_lcs))))

    stats.append(('Bytecode OpenCL source line counts',
                  seq_stats(bc_ocl_lcs)))
    stats.append(('Bytecode line counts',
                  seq_stats(bc_lcs)))
    stats.append(('',''))

    # CL tidy
    c.execute('SELECT contents FROM OpenCLTidy')
    cltidy = c.fetchall()

    nb_cltidy_files = len(cltidy)
    ratio_cltidy_files = nb_cltidy_files / nb_uniq_ocl_files
    stats.append(('Number of tidy OpenCL files',
                  bigint(nb_cltidy_files) +
                  ' ({:.0f}%)'.format(ratio_cltidy_files * 100)))

    cltidy_lcs = [len(x[0].split('\n')) for x in cltidy]
    cltidy_lcs.sort()
    stats.append(('Total line count of tidy OpenCL sources',
                  bigint(sum(cltidy_lcs))))

    stats.append(('Tidy OpenCL line counts',
                  seq_stats(cltidy_lcs)))
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

    c.execute('SELECT stars FROM ContentFiles LEFT JOIN Repositories '
              'ON ContentFiles.repo_url=Repositories.url')
    stars = [x[0] for x in c.fetchall()]

    # Filter range
    data = [x for x in stars if x < 100]

    sns.distplot(data, bins=20, kde=False)
    plt.xlabel('GitHub Stargazer count')
    plt.ylabel('Number of files')
    plt.title('Stargazers per file')
    plt.savefig(out_path)


def main():
    locale.setlocale(locale.LC_ALL, 'en_GB')

    if len(sys.argv) != 2:
        usage()
        sys.exit(1)

    db_path = sys.argv[1]

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # Worker process pool
    pool, jobs = Pool(processes=4), []
    jobs.append(pool.apply_async(graph_ocl_lc, (db_path,)))
    jobs.append(pool.apply_async(graph_bc_lc, (db_path,)))
    jobs.append(pool.apply_async(graph_ocl_stars, (db_path,)))

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
