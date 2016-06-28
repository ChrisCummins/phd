# (c) 2007 The Board of Trustees of the University of Illinois.

import sys
import os
from os import path
import re
from itertools import imap, repeat, chain

import globals
import process
import parboilfile as pbf
from futures import Future

from error import ErrorType

class Benchmark(object):
    """A benchmark.

    If the benchmark is malformed or otherwise invalid, only the 'name' and
    'invalid' fields will be set.  Otherwise all fields will be set.

    Fields:
      name        The name of the benchmark.  This is also the benchmark
                  directory name.
      invalid     None if the benchmark is valid; otherwise, an exception
                  describing why the benchmark is invalid.
      path        Full path of the benchmark directory.
      descr       A description of the benchmark.
      impls       A dictionary of benchmark source implementations.
      datas       A dictionary of data sets used to run the benchmark."""

    def __init__(self, name, path = None, impls = [], datasets = [],
                 description=None, invalid=None):
        self.name = name
        self.invalid = invalid

        if invalid is None:
            self.path = path
            self.impls = dict(imap(lambda i: (i.name, i), impls))
            self.datas = dict(imap(lambda i: (i.name, i), datasets))
            self.descr = description

    def createFromName(name):
        """Scan the benchmark directory for the benchmark named 'name'
        and create a benchmark object for it."""
        bmkdir = globals.benchdir.getChildByName(name)
        datadir = globals.datadir.getChildByName(name)
        descr = process.read_description_file(bmkdir)


        try:
            # Scan implementations of the benchmark
            impls = [BenchImpl.createFromDir(impl)
                     for impl in process.scan_for_benchmark_versions(bmkdir)]
            
            # Scan data sets of the benchmark
            datas = [BenchDataset.createFromDir(data)
                     for data in process.scan_for_benchmark_datasets(datadir)]

            # If no exception occurred, the benchmark is valid
            return Benchmark(name, bmkdir.getPath(), impls, datas, descr)
        finally:
            pass
        #except Exception, e:
        #    return Benchmark(name, invalid=e)
        
    createFromName = staticmethod(createFromName)

    def describe(self):
        """Return a string describing this benchmark."""

        if self.invalid:
            return "Error in benchmark:\n" + str(self.invalid)

        if self.descr  is None:
            header = "Benchmark '" + self.name + "'"
        else:
            header = self.descr

        impls = " ".join([impl.name for impl in self.impls.itervalues()])
        datas = " ".join([data.name for data in self.datas.itervalues()])

        return header + "\nVersions: " + impls + "\nData sets: " + datas

    def instance_check(x):
        if not isinstance(x, Benchmark):
            raise TypeError, "argument must be an instance of Benchmark"

    instance_check = staticmethod(instance_check)

class BenchImpl(object):
    """An implementation of a benchmark."""

    def __init__(self, dir, description=None):
        if not isinstance(dir, pbf.Directory):
            raise TypeEror, "dir must be a directory"

        self.name = dir.getName() 
        self.dir = dir
        self.descr = description

    def createFromDir(dir):
        """Scan the directory containing a benchmark implementation
        and create a BenchImpl object from it."""

        # Get the description from a file, if provided
        descr = process.read_description_file(dir)

        return BenchImpl(dir, descr)

    createFromDir = staticmethod(createFromDir)

    def makefile(self, benchmark, target=None, action=None, platform=None, opt={}):
        """Run this implementation's makefile."""
        
        self.platform = platform
        Benchmark.instance_check(benchmark)

        def perform():
            srcdir = path.join('src', self.name)
            builddir = path.join('build', self.name)

            if self.platform == None: platform = 'default'
            else: platform = self.platform

            env={'SRCDIR':srcdir,
                 'BUILDDIR':builddir + '_' + platform,
                 'BIN':path.join(builddir+'_'+platform,benchmark.name),
                 'PARBOIL_ROOT':globals.root,
                 'PLATFORM':platform,
                 'BUILD':self.name}
            env.update(opt)

            mkfile = globals.root + os.sep + 'common' + os.sep + 'mk'

            # Run the makefile to build the benchmark
            ret = process.makefile(target=target,
				    action=action,
                                    filepath=path.join(mkfile, "Makefile"),
                                    env=env)
            if ret == True:
              return ErrorType.Success
            else:
              return ErrorType.CompileError

        # Go to the benchmark directory before building
        return process.with_path(benchmark.path, perform)

    def build(self, benchmark, platform):
        """Build an executable of this benchmark implementation."""
        return self.makefile(benchmark, action='build', platform=platform)

    def isBuilt(self, benchmark, platform):
        """Determine whether the executable is up to date."""
        return self.makefile(benchmark, action='q', platform=platform) == ErrorType.Success

    def clean(self, benchmark, platform):
        """Remove build files for this benchmark implementation."""
        return self.makefile(benchmark, action='clean', platform=platform)

    def run(self, benchmark, dataset, do_output=True, extra_opts=[], platform=None):
        """Run this benchmark implementation.

        Return True if the benchmark terminated normally or False
        if there was an error."""

        if platform == None:
            self.platform = 'default'
        else:
            self.platform = platform

        # Ensure that the benchmark has been built
        if not self.isBuilt(benchmark, platform):
            rc = self.build(benchmark, platform)

            # Stop if 'make' failed
            if rc != ErrorType.Success: return rc

        def perform():
            if self.platform == None:
                platform = 'default'
            else:
                platform = self.platform

            # Run the program
            #exename = path.join('build', self.name+'_'+platform, benchmark.name)
            #args = [exename] + extra_opts + dataset.getCommandLineArguments(benchmark, do_output)
            #rc = process.spawnwaitv(exename, args)

            args = extra_opts + dataset.getCommandLineArguments(benchmark, do_output)
            args = reduce(lambda x, y: x + ' ' + y, args)

            ###
            try:
              rc = self.makefile(benchmark, action='run', platform=platform, opt={"ARGS":args})
            except KeyboardInterrupt:
              rc = ErrorType.Killed

            # Program exited with error?
            # if rc != 0: return ErrorType.RunFailed
            # return ErrorType.Success
            return rc

        return process.with_path(benchmark.path, perform)

    def debug(self, benchmark, dataset, do_output=True, extra_opts=[], platform=None):
        """Debug this benchmark implementation."""

        if platform == None:
            self.platform = 'default'
        else:
            self.platform = platform

        # Ensure that the benchmark has been built
        if not self.isBuilt(benchmark, platform):
            rc = self.build(benchmark, platform)

            # Stop if 'make' failed
            if rc != ErrorType.Success: return rc

        def perform():
            if self.platform == None:
                platform = 'default'
            else:
                platform = self.platform

            # Run the program
            args = extra_opts + dataset.getCommandLineArguments(benchmark, do_output)
            args = reduce(lambda x, y: x + ' ' + y, args)

            ###
            rc = self.makefile(benchmark, action='debug', platform=platform, opt={"ARGS":args})

            # Program exited with error?
            if rc != 0: return ErrorType.RunFailed
            return ErrorType.Success

        return process.with_path(benchmark.path, perform)

    def check(self, benchmark, dataset):
        """Check the output from the last run of this benchmark
        implementation.

        Return True if the output checks successfully or False
        otherwise."""

        def perform():
            output_file = dataset.getTemporaryOutputFile(benchmark).getPath()
            reference_file = dataset.getReferenceOutputPath()

            compare = os.path.join('tools', 'compare-output')
            rc = process.spawnwaitl(compare,
                                    compare, reference_file, output_file)

            # Program exited with error, or mismatch in output?
            if rc != 0: return False
            return True

        return process.with_path(benchmark.path, perform)

    def __str__(self):
        return "<BenchImpl '" + self.name + "'>"

class BenchDataset(object):
    """Data sets for running a benchmark."""

    def __init__(self, dir, in_files=[], out_files=[], parameters=[],
                 description=None):
        if not isinstance(dir, pbf.Directory):
            raise TypeError, "dir must be a pbf.Directory"

        self.name = dir.getName()
        self.dir = dir
        self.inFiles = in_files
        self.outFiles = out_files
        self.parameters = parameters
        self.descr = description

    def createFromDir(dir):
        """Scan the directory containing a dataset
        and create a BenchDataset object from it."""

        # Identify the paths where files may be found
        input_dir = dir.getChildByName('input')
        output_dir = dir.getChildByName('output')
        #benchmark_path = path.join(globals.root, 'benchmarks', name)

        
        def check_default_input_files():
            # This function is called to see if the input file set
            # guessed by scanning the input directory can be used
            if invalid_default_input_files:
                raise ValueError, "Cannot infer command line when there are multiple input files in a data set\n(Fix by adding an input DESCRIPTION file)"
                
        if input_dir.exists():
            input_descr = process.read_description_file(input_dir)
            input_files = input_dir.scanAndReturnNames()
            # If more than one input file was found, cannot use the default
            # input file list produced by scanning the directory
            invalid_default_input_files = len(input_files) > 1
        else:
            # If there's no input directory, assume the benchmark
            # takes no input
            input_descr = None
            input_files = []
            invalid_default_input_files = False

        # Read the text of the input description file
        if input_descr is not None:
            (parameters, input_files1, input_descr) = \
                unpack_dataset_description(input_descr, input_files=None)

            if input_files1 is None:
                # No override value given; use the default
                check_default_input_files()
            else:
                input_files = input_files1
        else:
            check_default_input_files()
            parameters = []

        # Look for output files
        output_descr = process.read_description_file(output_dir)
        output_files = output_dir.scanAndReturnNames()
        if len(output_files) > 1:
            raise ValueError, "Multiple output files not supported"

        # Concatenate input and output descriptions
        if input_descr and output_descr:
            descr = input_descr + "\n\n" + output_descr
        else:
            descr = input_descr or output_descr

        return BenchDataset(dir, input_files, output_files, parameters, descr)

    createFromDir = staticmethod(createFromDir)

    def getName(self):
        """Get the name of this dataset."""
        return self.name

    def getTemporaryOutputDir(self, benchmark):
        """Get the pbf.Directory for the output of a benchmark run.
        This function should always return the same pbf.Directory if its parameters
        are the same.  The output path is not the path where the reference
        output is stored."""

        rundir = globals.benchdir.getChildByName(benchmark.name).getChildByName('run')

        if rundir.getChildByName(self.name) is None:
            datasetpath = path.join(rundir.getPath(), self.name)
            filepath = path.join(datasetpath, self.outFiles[0])
            rundir.addChild(pbf.Directory(datasetpath, [pbf.File(filepath, False)]))
        
        return rundir.getChildByName(self.name)

    def getTemporaryOutputFile(self, benchmark):
        """Get the pbf.File for the output of a benchmark run.
        This function should always return the same pbf.File if its parameters 
        are the same.  The output path is not where the referrence output 
        is stored."""

        return self.getTemporaryOutputDir(benchmark).getChildByName(self.outFiles[0])


    def getReferenceOutputPath(self):
        """Get the name of the reference file, to which the output of a
        benchmark run should be compared."""

        return path.join(self.dir.getPath(), 'output', self.outFiles[0])

    def getCommandLineArguments(self, benchmark, do_output=True):
        """Get the command line arguments that should be passed to the
        executable to run this data set.  If 'output' is True, then
        the executable will be passed flags to save its output to a file.

        Directories to hold ouptut files are created if they do not exist."""
        args = []

        # Add arguments to pass input files to the benchmark
        if self.inFiles:
            in_files = ",".join([path.join(self.dir.getPath(),'input', x)
                                 for x in self.inFiles])
            args.append("-i")
            args.append(in_files)

        # Add arguments to store the output somewhere, if output is
        # desired
        if do_output and self.outFiles:
            if len(self.outFiles) != 1:
                raise ValueError, "only one output file is supported"

            out_file = self.getTemporaryOutputFile(benchmark)
            args.append("-o")
            args.append(out_file.getPath())

            # Ensure that a directory exists for the output
            self.getTemporaryOutputDir(benchmark).touch()

        args += self.parameters
        return args

    def __str__(self):
        return "<BenchData '" + self.name + "'>"

def unpack_dataset_description(descr, parameters=[], input_files=[]):
    """Read information from the raw contents of a data set description
    file.  Optional 'parameters' and 'input_files' arguments may be
    given, which will be retained unless overridden by the description
    file."""
    leftover = []
    split_at_colon = re.compile(r"^\s*([a-zA-Z]+)\s*:(.*)$")

    # Initialize these to default empty strings
    parameter_text = None
    input_file_text = None
    
    # Scan the description line by line
    for line in descr.split('\n'):
        m = split_at_colon.match(line)
        if m is None: continue

        # This line appears to declare something that should be
        # interpreted
        keyword = m.group(1)
        if keyword == "Parameters":
            parameter_text = m.group(2)
        elif keyword == "Inputs":
            input_file_text = m.group(2)
        # else, ignore the line

    # Split the strings into (possibly) multiple arguments, discarding
    # whitespace
    if parameter_text is not None: parameters = parameter_text.split()
    if input_file_text is not None: input_files = input_file_text.split()
    return (parameters, input_files, descr)

def version_scanner():
    """version_scanner() -> (path -> pbf.Directory) 
    
    Return a function to find benchmark versions in the src 
    directory for the benchmark."""

    return lambda x: pbf.scan_file(x, True, lambda y: pbf.Directory(y), ['.svn'])

def find_benchmarks():
    """Find benchmarks in the repository.  The benchmarks are
    identified, but their contents are not scanned immediately.  A
    dictionary is returned mapping benchmark names to futures
    containing the benchmarks."""

    if not globals.root:
        raise ValueError, "root directory has not been set"

    # Scan all benchmarks in the 'benchmarks' directory and
    # lazily create benchmark objects.
    db = {}

    try:
        globals.benchdir.scan()
        globals.datadir.scan()
        for bmkdir in globals.benchdir.getScannedChildren():
            bmk = Future(lambda bmkdir=bmkdir: Benchmark.createFromName(bmkdir.getName()))
            db[bmkdir.getName()] = bmk
    except OSError, e:
        sys.stdout.write("Benchmark directory not found!\n\n")
        return {}

    return db

def _desc_file(dpath):
    """_desc_file(dpath) 
    Returns a pbf.File for an optional description file in the directory dpath."""

    return pbf.File(path.join(dpath,'DESCRIPTION'), False)

def benchmark_scanner():
    """benchmark_scanner -> (path -> pbf.Directory)

    Returns a function which will scan a filename and create a pbf.Directory 
    for a benchmark represented by that name."""

    def create_benchmark_dir(dpath):
        expected = [pbf.Directory(path.join(dpath,'src'), [], version_scanner()),
                    pbf.Directory(path.join(dpath,'tools'), 
                              [pbf.File(path.join(dpath,'compare-output'))]),
                    pbf.Directory(path.join(dpath,'build'), must_exist=False),
                    pbf.Directory(path.join(dpath,'run'), must_exist=False),
                    _desc_file(dpath)]
        return pbf.Directory(dpath, expected)

    return lambda x: pbf.scan_file(x, True, create_benchmark_dir,['_darcs','.svn'])

def dataset_scanner():
    """dataset_scanner -> (path -> pbf.Directory)

    Returns a function which will scan a filename and create a pbf.Directory
    for a folder containing datasets for the benchmark of the same name."""

    def create_dataset_dir(dpath):
        simple_scan = lambda x: pbf.scan_file(x)
        expected = [pbf.Directory(path.join(dpath,'input'), 
                              [_desc_file(path.join(dpath,'input'))], simple_scan),
                    pbf.Directory(path.join(dpath,'output'), [], simple_scan),
                    _desc_file(dpath)]

        return pbf.Directory(dpath, expected)

    return lambda x: pbf.scan_file(x, True, create_dataset_dir, ['.svn', '_darcs'])

def dataset_repo_scanner():
    """dataset_repo_scanner -> (path -> pbf.Directory)

    Returns a function which will scan a filename and create a pbf.Directory 
    for a folder containing a dataset repository for parboil benchmarks."""

    benchmark_dsets_scanner = lambda x: pbf.Directory(x, [], dataset_scanner())

    return lambda x: pbf.scan_file(x, True, benchmark_dsets_scanner)
