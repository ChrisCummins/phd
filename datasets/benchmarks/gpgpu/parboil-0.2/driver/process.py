# (c) 2007 The Board of Trustees of the University of Illinois.

# Process-management and directory management routines are collected here.

import os
import os.path as path
import stat
import parboilfile as pbf
from itertools import imap, ifilter, chain

import globals

#def scan_for_benchmarks():
#    """Returns a file scanner for the benchmarks directory repository to find
#    subdirectories signifying benchmarks."""


    #expecteddirs = [Directory(path.join(root,x'benchmarks','build', 'input', 'output', 'run', 'rools', 'src')]

#    return lambda x: scan_file(x, True, lambda x: Directory(, boring=['_darcs','.svn'])
        
def scan_for_benchmark_versions(bmkdir):
    """Scan subdirectories of a benchmark Directory 'bmkdir' to find
    benchmark versions.  Return a sequence containing all benchmark
    version names."""

    srcdir = bmkdir.getChildByName('src')
    srcdir.scan()
    return srcdir.getScannedChildren()

def scan_for_benchmark_datasets(bmkdir):
    """Scan subdirectories of a benchmark directory 'bmkdir' to find
    data sets.  Return a sequence containing all data set names."""

    bmkdir.scan()
    return bmkdir.getScannedChildren()

def read_description_file(dir):
    """Read the contents of a file in Directory 'dir' called DESCRIPTION,
    if one exists.  This returns the file text as a string, or None
    if no description was found."""

    descr_handle = dir.getChildByName('DESCRIPTION')
    if descr_handle is not None:
        if os.access(descr_handle.getPath(), os.R_OK):
            descr_file = descr_handle.open()
            descr = descr_file.read()
            descr_file.close()
            return descr
    
    # else, return None

def with_path(wd, action):
    """Executes an action in a separate working directory.  The action
    should be a callable object."""
    cwd = os.getcwd()
    os.chdir(wd)
    try: result = action()
    finally: os.chdir(cwd)
    return result
    
def makefile(target=None, action=None, filepath=None, env={}):
    """Run a makefile.  An optional command, makefile path, and dictionary of
    variables to define on the command line may be defined.  The return code
    value is the return code returned by the makefile.

    If no action is given, 'make' is invoked.  Returns True if make was
    successful and False otherwise.

    A 'q' action queries whether the target needs to be rebuilt.  True is
    returned if the target is up to date."""

    args = ["make"]

    if action is 'build':
        def run():
            args.append('default')
            rc = os.spawnvp(os.P_WAIT, "make", args)
            return rc == 0
    elif action is 'clean':
        def run():
            args.append('clean')
            rc = os.spawnvp(os.P_WAIT, "make", args)
            return rc == 0
    elif action is 'run':
        def run():
            args.append('run')
            rc = os.spawnvp(os.P_WAIT, "make", args)
            return rc == 0
    elif action is 'debug':
        def run():
            args.append('debug')
            rc = os.spawnvp(os.P_WAIT, "make", args)
            return rc == 0
    elif action in ['q']:
        args.append('-q')

        def run():
            rc = os.spawnvp(os.P_WAIT, "make", args)
            if rc == 0:
                # Up-to-date
                return True
            elif rc == 1:
                # Needs remake
                return False
            else:
                # Error
                return False
    else:
        raise ValueError, "invalid action"

    # Pass the target as the second argument
    if target: args.append(target)

    # Pass the path the the makefile
    if filepath:
        args.append('-f')
        args.append(filepath)

    # Pass variables
    for (k,v) in env.iteritems():
        args.append(k + "=" + v)

    # Print a status message, if running in verbose mode
    if globals.verbose:

        print "Running '" + " ".join(args) + "' in " + os.getcwd()

    # Run the makefile and return result info
    return run()

def spawnwaitv(prog, args):
    """Spawn a program and wait for it to complete.  The program is
    spawned in a modified environment."""

    env = dict(os.environ)
    env.update(globals.program_env)

    # Print a status message if running in verbose mode
    if globals.verbose:
        print "Running '" + " ".join(args) + "' in " + os.getcwd()

    # Check that the program is runnable
    if not os.access(prog, os.X_OK):
        raise OSError, "Cannot execute '" + prog + "'"

    # Run the program
    return os.spawnve(os.P_WAIT, prog, args, env)

def spawnwaitl(prog, *argl):
    """Spawn a program and wait for it to complete.  The program is
    spawned in a modified environment."""

    return spawnwaitv(prog, argl)
