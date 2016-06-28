#!/usr/bin/env python

import re
import os
import sys
import tarfile

import os
from os.path import join, getsize

class Archiver:
  def __init__(self, name, dont_run=False):
    self.filters = []
    self.benchmarks = set([])
    self.dont_run = dont_run
    if not dont_run:
      self.tarfile = tarfile.open(name, 'w:gz')

  def readFilter(self):
    try:
      f = open('ppkg.ignore', 'r') 
    except:
      "No filters."
      return

    for l in f.readlines():
      l = l.strip()
      if l == '': continue
      if l.startswith('#'): continue
      try:
        self.filters.append(re.compile(l))
      except:
        print "Not valid pattern: %s" % l
    f.close()

  def readBenchmarks(self):
    try:
      f = open('ppkg.benchmarks', 'r')
    except:
      "No benchmarks."
      return

    for l in f.readlines():
      l = l.strip()
      if l == '': continue
      if l.startswith('#'): continue
      self.benchmarks.add(l)
    f.close()

  def ignore(self, name):
    result = None
    for f in self.filters:
      if f.search(name) == None: continue
      return True
    return False

  def run(self, top):
    (path, dir) = os.path.split(top)
    self.top = top
    os.chdir(path)
    if (len(self.benchmarks) == 0):
    	self.walk(dir)
    else:
      self.walk_benchmarks(dir)
    self.close()

  def walk(self, top):
    for root, dirs, files in os.walk(top):
      for dir_ in dirs:
        self.walk(dir_)
      for file in files:
        fullName = root + os.sep + file
        if not self.ignore(fullName):
          self.add(fullName)
        else:
          print "Ignore %s" % fullName

  def walk_benchmarks(self, top):
    for entry in os.listdir(top):
    	if entry in self.benchmarks:
    		self.walk(os.path.join(top, entry))
    	else:
    		print "Ignoring %s" % entry

  def add(self, path):
    print "Adding %s ..." % path
    if not self.dont_run:
      self.tarfile.add(path)

  def close(self):
    if not self.dont_run:
      self.tarfile.close()

if len(sys.argv) != 3:
  print "Usage: ./ppkg.py <output filename WITH extension: XXX.tar.gz> <input directory>"
  print "       add your own ignore list in pkg.ignore file in the current directory"

a = Archiver(sys.argv[1], False)
a.readFilter()
a.readBenchmarks()
a.run(sys.argv[2])

