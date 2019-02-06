#!/usr/bin/env python
#
# Parboil sanity checker
#
# This script does the following:
# 1. check out up-to-date copy of parboil
# 2. run all possible combinations of benchmarks, versions, and inputs
# 3. create result-<date>.txt which summarizes the test
#
# How to run this script:
# 1. make sure you run this script on a system which has access to AFS
#    specifically, it should have an access to
#      /afs/crhc.illinois.edu/projects/hwuligan/parboil
#    , which is usually the case with most of the lab machines
# 2. make an empty directory
# 3. copy this script file into the directory (NOT THE ENTIRE REPOSITORY)
# 4. run the script by typing: "python sanity_check.py"
# 5. wait (hours, maybe a day, not a week long though)
# 6. you will get the result. 
#
# questions? kim868@illinois.edu
#

import shutil
import os
import pdb
import time
import subprocess
import socket

err_string = {
  0: 'success',
  1: 'compile error',
  2: 'runtime error',
  3: 'output mismatch',
  4: 'cannot find version',
  5: 'cannot find data set',
  6: 'killed by user'
}

class benchmark:
  pbTimerCats = [
    "IO        : ",
    "GPU       : ",
    "Copy      : ",
    "Driver    : ",
    "Copy Async: ",
    "Compute   : ",
    "CPU/GPU Overlap: ",
    "Timer Wall Time:",
    ]

  def __init__(self, name, platforms):
    self.name = name
    self.vers = []
    self.data = []
    self.platforms = platforms

    self.results = []

    self.scan()

  def scan(self):
    self.scan_versions()
    self.scan_inputs()

  def scan_versions(self):
    for ver in os.listdir('benchmarks' + os.sep + self.name + os.sep + 'src'):
      self.vers.append(ver) 

  def scan_inputs(self):
    for datum in os.listdir('datasets' + os.sep + self.name):
      self.data.append(datum) 

  def run(self, ver, datum, pl, fake=False): 
    cmd = 'python parboil run %s %s %s %s' % (self.name, ver, datum, pl)
    print cmd

    if fake: return 0, ''

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

    timing = ''
    while True:
      line = p.stdout.readline()
      # if line == '' and p.poll() != None: break
      if line == '': break

      isCat = False
      for cat in self.pbTimerCats:
        if line.find(cat) != -1:
          line = " ++ %s" % line
          print line,
          timing += line
          isCat = True
          break
      if not isCat: print line,

    sts = os.waitpid(p.pid, 0)[1]
    # return p.returncode, timing
    return sts, timing

  def run_all(self, fake=False):
    for ver in self.vers:
      if ver in [ 'cpu', 'base' ]:
        continue

      else:
        for datum in self.data:
          for platform in self.get_platform(ver):
            ret, timing = self.run(ver, datum, platform, fake)

            # FIXME
            # Stupid something in between system() call and the actual process
            # Don't know why but sometimes the value is left shifted by 8 bits
            if ret >= 256: ret >>= 8

            self.results.append((ver, datum, platform, ret, timing))

  def get_results(self): 
    return self.results

  def get_result_string(self):
    #ss = "%s: %s versions, %s test cases\n" % \
    #      (self.name, len(self.vers), len(self.data))
    ss = ''
    for result in self.results:
      ver, datum, platform, ret, timing = result
      try:
        ss += "%s: %s: %s: %s: %s\n" % \
              (self.name, ver, datum, platform, err_string[ret])
      except:
        ss += "%s: %s: %s: %s: unknown error\n" % \
              (self.name, ver, datum, platform)

      if ret == 0:
        ss += timing
    return ss

  def write_result(self, out):
    x = self.get_result_string()
    print x,
    out.write(x)
    out.flush()

  def get_platform(self, ver):
    """adhoc Makefile reader for LANGUAGE variable"""

    makefile = 'benchmarks' + os.sep + \
                self.name + os.sep + \
                'src' + os.sep + \
                ver + os.sep + 'Makefile'

    h = open(makefile, 'r')
    ls = h.readlines()
    h.close()
    for l in ls:
      if l.find('#') < l.find('LANGUAGE='):
        lang, val = l.split('=')
        pl = val.split('#')[0].strip()
        return self.platforms[pl]

def checkout():
  parboil_repos = '/afs/crhc.illinois.edu/project/hwuligans/parboil/parboil'
  bmks_repos = '/afs/crhc.illinois.edu/project/hwuligans/parboil/benchmarks'
  datasets_path = '/afs/crhc.illinois.edu/project/hwuligans/parboil/datasets/'

  os.system('rm -rf _parboil')
  os.system('darcs get %s _parboil' % parboil_repos)
  os.system('darcs get %s _parboil/benchmarks' % bmks_repos)
  os.system('cd _parboil && ln -s %s' % datasets_path)

def prepare():
  os.system('PARBOIL_ROOT=$PWD/_parboil && cd _parboil/common/src/ && make')
  os.system('cd _parboil && find . -name "compare-output" -exec chmod +x {} \\;')
  os.system('chmod +x _parboil/parboil')

def init():
  out = get_output_file()

  pwd = os.getcwd()
  os.chdir('_parboil')
  os.environ['PARBOIL_ROOT'] = os.getcwd()

  return pwd, out

def get_benchmarks(platform):
  platforms = scan_platforms(platform)

  bmks = []
  for d in os.listdir('benchmarks'):
    if d in [ '.', '_darcs' ]:
     continue
    bmks.append(benchmark(d, platforms))
  return bmks

def scan_platforms(platform):
  path = 'common/platform'

  platforms = {}
  for mks in os.listdir(path):
    if mks[-3:] == '.mk':

      file = path + os.sep + mks
      if os.path.islink(path + os.sep + mks): continue

      tokens = mks.split('.')
      lang = tokens[0]
      if lang not in platforms.keys():
        platforms[lang] = []
      platforms[lang].append(reduce(lambda x, y: x+y, tokens[1:-1]))
  return platforms

def get_output_file():
  fname = 'result-%s.txt' % time.strftime('%Y-%m-%d', time.gmtime())
  return open(fname, 'w')

def run(platform):
  shutil.copy('_parboil/common/Makefile.conf.example-%s' % platform, '_parboil/common/Makefile.conf')

  pwd, out = init()
  bmks = get_benchmarks(platform)

  for bmk in bmks:
    bmk.run_all(fake=False)
    bmk.write_result(out)

  out.close()
  os.chdir(pwd)

###########################################

checkout()
prepare()
if socket.gethostname().startswith('cyclone'):
  run('nvidia')
elif socket.gethostname().startswith('ati'):
  run('ati')

