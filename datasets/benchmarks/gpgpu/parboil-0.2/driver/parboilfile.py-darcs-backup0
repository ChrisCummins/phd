# File objects used by the parboil driver.
#
# A File holds information about a file or directory that the Parboil driver
# may interact with.  In general, the file or direcotry may or may not be
# open, and may or may not exist.
# File objects contain utility routines to help the driver scan directories
# and recover from errors.

# This code replaces some of the functionality of the following functions:
#   scan_for_files
#   scan_for_benchmarks
#   scan_for_benchmark_versions
#   scan_for_benchmark_datasets
#   touch_directory
#   find_benchmarks
#   Benchmark.createFromName
#   BenchImpl.createFromName
#   BenchDataset.createFromName
#   some of the os.path calls in benchmark.py

# Each Benchmark should reference a Directory object for its code and
# a Directory object for its shared data.
# Each BenchImpl should reference a Directory object.
# Each BenchDataset should reference a Directory object for the data set.

import os
import os.path as path

class FileBase(object):
    """The base class for file objects.  An instance of FileBase has
    information about one file or directory.

    A FileBase instance reflects what is *expected* about the corresponding
    file in the file system.  The actual state of the file system might be
    different.  For example, a file described by a FileBase object might not
    actually exist."""

    def valid(self):
        """f.valid() -> bool

        Test whether this file is valid.  If the file doesn't exist and it
        is not required to exist, then it is valid.  If the file exists,
        then ths function returns the same results as exists().  This function
        should not raise an exception."""
        raise NotImplementedError, "'FileBase' is an abstract base class"

    def exists(self):
        """f.exists() -> bool

        Test whether this file exists and is a valid file.  This function
        should not raise an exception."""
        raise NotImplementedError, "'FileBase' is an abstract base class"

    def isDir(self):
        """f.isDir() -> bool

        Test whether this FileBase object represents a directory.  This
        function should not access the file system."""
        raise NotImplementedError, "'FileBase' is an abstract base class"

    def isFile(self):
        """f.isFile() -> bool

        Test whether this FileBase object represents an ordinary file.  This
        function should not access the file system."""
        raise NotImplementedError, "'FileBase' is an abstract base class"

    def getPath(self):
        """f.getPath() -> string

        Get the path to this file.  This function should not access the
        file system."""
        raise NotImplementedError, "'FileBase' is an abstract base class"

    def getName(self):
        """f.getName() -> string

        Get the name of this file.  This function should not access the 
        file system."""
        raise NotImplementedError, "'FileBase' is an abstract base class"

class File(FileBase):
    """A description of a file."""
    def __init__(self, fpath, must_exist = True):
        """File(path, must_exist)

        Create a description of a file, containing its path and whether or 
        not it must exist.  This function does not access the file system."""

        self._path = fpath
        self._must_exist = must_exist
        self._name = path.split(fpath)[1]

    def exists(self):
    	try: return path.isfile(self.getPath())
    	except OSError: return False # handles file-not-found case

    def valid(self):
    	if self._must_exist:
    	    return self.exists()
    	else:
    	    return True

    def isDir(self): return False

    def isFile(self): return True

    def getPath(self):
    	return self._path

    def getName(self):
        return self._name

    def open(self, mode='r', buffering=None):
        """f.open(mode, buffering) -> file object

        Open the file."""
        if buffering is None: return file(self.getPath(), mode)
        else: return file(self.getPath(), mode, buffering)

class Directory(FileBase):
    """A description of a directory."""
    def __init__(self, dpath,
                 contents_list = [], scan_func = None, must_exist = True):
        """Directory(path, contents-list, scan-function or None, must-exist)

        Create a Directory object.  Files given as part of contents_list
        are added to directory.  Any other files in the directory are
        passed (by full path) to the scanner function to decide whether to 
        include them.  The scanner function should return None if a file is 
        to be ignored or a FileBase object if the file is to be noticed.

        This function does not access the file system."""
        for f in contents_list:
            assert isinstance(f, FileBase)
        if scan_func and not must_exist:
            raise ValueError, "Invalid combination of arguments: scan_func is provided but must_exist is False"

        self._realContents = None
        self._interesting = scan_func
        self._mustExist = must_exist
        self._contentsList = contents_list
        self._path = dpath
        self._name = path.split(dpath)[1]

    def exists(self):
    	try: return path.isdir(self.getPath())
    	except OSError: return False # handles file-not-found case

    def valid(self):

    	if self._mustExist and not self.exists():
    	    return False
    	
    	if self.exists():
    	    for file in self._contentsList:
    	        if not file.valid(): return False
    	    if self._realContents is not None:
    	        for file in self._realContents:
    	            if not file.valid(): return False

        #Children are valid, and either exists or doesn't have to
        return True
    
    def isDir(self): return True

    def isFile(self): return False

    def getPath(self):
    	return self._path

    def getName(self):
        return self._name

    def scan(self):
        """d.scan() -- scan the contents of the file system to find the
        contents of this directory."""
        if self._realContents is not None: return
        
        # Scan the directory and assign self._realContents
        if not self.exists(): 
            raise OSError, "Directory '" + self._name + "' does not exist."
        
        all_contents = os.listdir(self.getPath())

	def has_file_of_name(name):
	    for x in self._contentsList:
	        if x.getName() == name:
	            return True
	    return False

        new_contents = filter(lambda x: not has_file_of_name(x), all_contents)

        self._realContents = filter(lambda x: x is not None, 
        	[self._interesting(path.join(self.getPath(), x)) for x in new_contents])

    def touch(self):
        """d.touch() -- create this directory if it doesn't exist.
        Throw an error if the file cannot be created, exists but is not a
        directory, or exists but is not readable, writable, and listable by
        the user."""
        def touch_dir(dirpath):
            """Ensures that the directory 'dirpath' and its parent directories
            exist.  If they do not exist, they will be created.  It is an
            error if the path exists but is not a directory."""
            if path.isdir(dirpath):
                return
            elif path.exists(dirpath):
                raise OSError, "Path exists but is not a directory"
            else:
                (head, tail) = path.split(dirpath)
                if head: touch_dir(head)
                os.mkdir(dirpath)

        touch_dir(self._path)

    def getChildByName(self, filename):
        """d.getChild(filename) -> FileBase object

        Get a file from this directory.
        Raise an exception if the file is not in this directory."""
        for x in self.getChildren():
            if x.getName() == filename:
                return x
        return None

    
    def getChildByPath(self, pathname):
        """d.getChild(filepath) -> FileBase object

        Get a file from this directory.  Verify that the file exists
        before returning it.  Raise an exception if the file doesn't exist
        or is ignored by this directory."""
        (basepathname, filename) = path.split(pathname)
        if basepathname != self.getPath():
            raise ValueError, "Path is not a child of this directory"
        return self.getChildByName(filename)

    def getChildren(self):
        """d.getChildren(filepath) -> list of FileBase

        Get a list of all children of this directory."""

        if self._realContents is None:
            return self._contentsList
        else:
            return self._contentsList + self._realContents

    def getScannedChildren(self):
        """d.getScannedChildren(filepath) -> list of FileBase

        Get a list of all children of this directory that were found by
        scanning the directory.  Explicitly listed contents are not
        included in the list."""
        return self._realContents

    def scanAndReturnNames(self):
        """Exectues the scan operation for this directory, returning a list of 
        strings representing the names of files discovered."""

        self.scan()
        return [ x.getName() for x in self.getScannedChildren() ]

    def addChild(self, newchild):
        """addChild(newchild)
        Adds parameter to its expected children list.  Returns None.  
        Parameter must be a FileBase obejct."""
        if not isinstance(newchild, FileBase):
            raise TypeError, "parameter must be a FileBase object"

        self._contentsList.append(newchild)

def scan_file(fpath, directory=False, create_file=lambda x: File(x), boring=[]):
    """scan_file(path, directory=False, create_file, boring=[]) -> FileBase
    
    Test the file referenced by path.  Invoke creat_file on that path 
    if it is to be noticed and returns the results, else returns None.  
    A file is considered 'to be noticed' if its name is not in boring, 
    and it is a directory or file if directory is True or False, respectively. 
    This function accesses the file system to determine the file type 
    (directory or file) and raises an OSError on inability to access.  
    """

    # Look for directories or regular files, depending on the 'directory'
    # parameter
    if directory: valid_test = path.isdir
    else: valid_test = path.isfile
    
    # True if 'dirname' is not a boring name, and it is a directory
    fname = path.split(fpath)[1]

    if fname in boring: return None

    if valid_test(fpath): return create_file(fpath)


