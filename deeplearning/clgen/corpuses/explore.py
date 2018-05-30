#
# Copyright 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
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
Exploratory analysis of OpenCL dataset
"""
import locale

from deeplearning.clgen import dbutil


IMG_DIR = 'img'


def _safe_div(x, y) -> float:
  """
  Zero-safe _safe_division.

  Parameters
  ----------
  x : Number
      Numerator.
  y : Number
      Denominator.

  Returns
  -------
  Number
      x / y
  """
  try:
    return x / y
  except ZeroDivisionError:
    return 0


def _bigint(n) -> str:
  """
  Return comma seperated big numbers.

  Parameters
  ----------
  n : Number
      Value.

  Returns
  -------
  str
      Comma separated value.
  """
  return locale.format('%d', round(n), grouping=True)


def _seq_stats(sorted_arr: list) -> str:
  """
  Return stats on a sequence.

  Parameters
  ----------
  sorted_arr : List[Number]
      Sequence.

  Returns
  -------
  str
      Sequnece stats.
  """
  sorted_arr = sorted_arr or [0]
  avg = sum(sorted_arr) / len(sorted_arr)
  midpoint = int(len(sorted_arr) / 2)
  if len(sorted_arr) % 2 == 1:
    median = sorted_arr[midpoint]
  else:
    median = (sorted_arr[midpoint - 1] + sorted_arr[midpoint]) / 2.0
  return ('min: {}, med: {}, avg: {}, max: {}'.format(_bigint(sorted_arr[0]),
                                                      _bigint(median),
                                                      _bigint(avg), _bigint(
        sorted_arr[len(sorted_arr) - 1])))


def explore(db_path: str) -> None:
  """
  Run exploratory analysis on dataset.

  Parameters
  ----------
  db_path : str
      Path to dataset.
  """
  locale.setlocale(locale.LC_ALL, 'en_GB.utf-8')

  db = dbutil.connect(db_path)

  if dbutil.is_github(db):
    db.close()
    explore_gh(db_path)
    return

  db = dbutil.connect(db_path)
  c = db.cursor()
  stats = []

  # ContentFiles
  c.execute("SELECT Count(DISTINCT id) from ContentFiles")
  nb_uniq_ocl_files = c.fetchone()[0]
  stats.append(('Number of content files', _bigint(nb_uniq_ocl_files)))

  c.execute("SELECT contents FROM ContentFiles")
  code = c.fetchall()
  code_lcs = [len(x[0].split('\n')) for x in code]
  code_lcs.sort()
  code_lc = sum(code_lcs)
  stats.append(('Total content line count', _bigint(code_lc)))

  stats.append(('Content file line counts', _seq_stats(code_lcs)))
  stats.append(('', ''))

  # Preprocessed
  c.execute("SELECT Count(*) FROM PreprocessedFiles")
  nb_pp_files = c.fetchone()[0]
  ratio_pp_files = _safe_div(nb_pp_files, nb_uniq_ocl_files)
  stats.append(('Number of preprocessed files',
                _bigint(nb_pp_files) + ' ({:.0f}%)'.format(
                    ratio_pp_files * 100)))

  c.execute("SELECT Count(*) FROM PreprocessedFiles WHERE status=0")
  nb_pp_files = c.fetchone()[0]
  ratio_pp_files = _safe_div(nb_pp_files, nb_uniq_ocl_files)
  stats.append(('Number of good preprocessed files',
                _bigint(nb_pp_files) + ' ({:.0f}%)'.format(
                    ratio_pp_files * 100)))

  c.execute('SELECT contents FROM PreprocessedFiles WHERE status=0')
  bc = c.fetchall()
  pp_lcs = [len(x[0].split('\n')) for x in bc]
  pp_lcs.sort()
  pp_lc = sum(pp_lcs)
  ratio_pp_lcs = _safe_div(pp_lc, code_lc)
  stats.append(('Lines of good preprocessed code',
                _bigint(pp_lc) + ' ({:.0f}%)'.format(ratio_pp_lcs * 100)))

  stats.append(('Good preprocessed line counts', _seq_stats(pp_lcs)))
  stats.append(('', ''))

  # Print stats
  print()
  maxlen = max([len(x[0]) for x in stats])
  for stat in stats:
    k, v = stat
    if k:
      print(k, ':', ' ' * (maxlen - len(k) + 2), v, sep='')
    elif v == '':
      print(k)
    else:
      print()


def explore_gh(db_path: str) -> None:
  """
  Run exploratory analysis on GitHub dataset.

  Parameters
  ----------
  db_path : str
      Path to dataset.
  """
  locale.setlocale(locale.LC_ALL, 'en_GB.utf-8')

  db = dbutil.connect(db_path)
  c = db.cursor()
  stats = []

  # Repositories
  c.execute("SELECT Count(*) from Repositories")
  nb_repos = c.fetchone()[0]
  stats.append(('Number of repositories visited', _bigint(nb_repos)))
  stats.append(('', ''))

  c.execute("SELECT Count(DISTINCT repo_url) from ContentMeta")
  nb_ocl_repos = c.fetchone()[0]
  stats.append(('Number of content file repositories', _bigint(nb_ocl_repos)))

  c.execute('SELECT Count(*) FROM Repositories WHERE fork=1 AND url IN '
            '(SELECT repo_url FROM ContentMeta)')
  nb_forks = c.fetchone()[0]
  ratio_forks = _safe_div(nb_forks, nb_ocl_repos)
  stats.append(('Number of forked repositories',
                _bigint(nb_forks) + ' ({:.0f}%)'.format(ratio_forks * 100)))

  # ContentFiles
  c.execute("SELECT Count(DISTINCT id) from ContentFiles")
  nb_uniq_ocl_files = c.fetchone()[0]
  ratio_uniq_ocl_files = _safe_div(nb_uniq_ocl_files, nb_uniq_ocl_files)
  stats.append(('Number of unique content files', _bigint(nb_uniq_ocl_files)))

  avg_nb_ocl_files_per_repo = _safe_div(nb_uniq_ocl_files, nb_ocl_repos)
  stats.append(('Content files per repository',
                'avg: {:.2f}'.format(avg_nb_ocl_files_per_repo)))

  c.execute("SELECT contents FROM ContentFiles")
  code = c.fetchall()
  code_lcs = [len(x[0].split('\n')) for x in code]
  code_lcs.sort()
  code_lc = sum(code_lcs)
  stats.append(('Total content line count', _bigint(code_lc)))

  stats.append(('Content file line counts', _seq_stats(code_lcs)))
  stats.append(('', ''))

  # Preprocessed
  c.execute("SELECT Count(*) FROM PreprocessedFiles")
  nb_pp_files = c.fetchone()[0]
  ratio_pp_files = _safe_div(nb_pp_files, nb_uniq_ocl_files)
  stats.append(('Number of preprocessed files',
                _bigint(nb_pp_files) + ' ({:.0f}%)'.format(
                    ratio_pp_files * 100)))

  c.execute("SELECT Count(*) FROM PreprocessedFiles WHERE status=0")
  nb_pp_files = c.fetchone()[0]
  ratio_pp_files = _safe_div(nb_pp_files, nb_uniq_ocl_files)
  stats.append(('Number of good preprocessed files',
                _bigint(nb_pp_files) + ' ({:.0f}%)'.format(
                    ratio_pp_files * 100)))

  c.execute('SELECT contents FROM PreprocessedFiles WHERE status=0')
  bc = c.fetchall()
  pp_lcs = [len(x[0].split('\n')) for x in bc]
  pp_lcs.sort()
  pp_lc = sum(pp_lcs)
  ratio_pp_lcs = _safe_div(pp_lc, code_lc)
  stats.append(('Lines of good preprocessed code',
                _bigint(pp_lc) + ' ({:.0f}%)'.format(ratio_pp_lcs * 100)))

  stats.append(('Good preprocessed line counts', _seq_stats(pp_lcs)))
  stats.append(('', ''))

  # Print stats
  print()
  maxlen = max([len(x[0]) for x in stats])
  for stat in stats:
    k, v = stat
    if k:
      print(k, ':', ' ' * (maxlen - len(k) + 2), v, sep='')
    elif v == '':
      print(k)
    else:
      print()
