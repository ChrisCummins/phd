"""This file implements the common library code of shutterbug.

Shutterbug is a library for creating DVD backups of photo libraries.
"""
import gzip
import os
import pathlib
import random
import sys
import typing
from datetime import datetime
from hashlib import md5
from shutil import copy
from shutil import copyfileobj

from labm8.py import app

FLAGS = app.FLAGS

# Metadata used in README.txt files.
__author__ = "Chris Cummins"
__version__ = "0.0.4"
__email__ = "chrisc.101@gmail.com"


def md5sum(path):
  m = md5()
  with open(path, "rb") as infile:
    m.update(infile.read())
  return m.hexdigest()


def chunk_files(
  paths, outdir, maxsize, prefix="chunk_", shuffle=True, seed=None, gzip=False
):
  """Create chunk files.

  Split a set of file paths into chunks, whereby the cumulative size of files
  in each chunk is smaller than or equal to a maximum size in bytes.

  Paths is a list of tuples, where each tuple consists of a path to a file,
  and the root directory for that file.

  Yields a list of tuples, where each tuple consists of path to the file, and
  its size in bytes.

  If shuffle is True, files are sorted into a random order prior to chunking.
  This can help distribute data loss in case a chunk becomes corrupted, and
  help minimize the number of chunks required in case there are groups of
  large files.

  Returns a list of tuples, where each tuple consists of a path to a chunk,
  and its size in bytes.
  """

  def chunk_meta(chunk_path, chunk, chunksize, maxsize):
    manifestpath = os.path.join(chunk_path, "MANIFEST.txt")
    with open(manifestpath, "w") as outfile:
      for outpath, path, size, checksum in sorted(chunk, key=lambda x: x[0]):
        print(outpath, checksum, size, path, file=outfile, sep="\t")
    print("Wrote", manifestpath)

    readmepath = os.path.join(chunk_path, "README.txt")
    progpath, version = __file__, __version__
    author, email = __author__, __email__
    date = datetime.now()
    with open(readmepath, "w") as outfile:
      print(
        """\
Created by: {progpath}
Author: {author} <{email}>
Version: {version}
Date: {date}

The MANIFEST.txt file contains a tab separated list of filenames, MD5 checksums,
file sizes, and original file paths. To restore the original files, for each
line in the manifest file:
  1. Copy (or unzip if file ends with .gz) the file path of the first column to
     the output file path in the fourth column.
  2. Compare the output file md5sum against the checksum in the second column.
  3. Compare the output file size against the file size of the third column.\
""".format(
          **vars()
        ),
        file=outfile,
      )
    print("Wrote", readmepath)

    chunksize_mb = chunksize / 1000 ** 2
    nfiles = len(chunk)
    chunksize_perc = (chunksize / maxsize) * 100
    print(
      "{chunk_path} has {nfiles} files, size {chunksize_mb:.2f} MB "
      "({chunksize_perc:.1f}% of maximum size)".format(**vars())
    )
    print()

  def get_outpath(chunk_path, path, gzip=False):
    checksum = md5sum(path)
    ext = os.path.splitext(path)[1]  # file extension

    if ext == ".gz":  # prevent double-zipping
      ext = "-gz"

    if gzip:
      outpath = os.path.join(chunk_path, checksum + ext) + ".gz"
    else:
      outpath = os.path.join(chunk_path, checksum + ext)

    # generate a unique file name
    if os.path.exists(outpath):
      i = 2
      while True:
        outpath = os.path.join(chunk_path, checksum + "-" + str(i) + ext)
        if not os.path.exists(outpath):
          return checksum, outpath
        i += 1
    else:
      return checksum, outpath

  def cp(path, size, chunk_path):
    checksum, outpath = get_outpath(chunk_path, path)

    copy(path, outpath)
    print(outpath, "{:.2f}MB".format(size / 1000 ** 2))
    return os.path.basename(outpath), checksum

  def gz(path, size, chunk_path):
    checksum, outpath = get_outpath(chunk_path, path, gzip=True)

    with open(path, "rb") as infile:
      with gzip.open(outpath, "wb") as outfile:
        copyfileobj(infile, outfile)
    outsize = os.stat(outpath).st_size

    if outsize <= size:
      # see if compressed file size is smaller than starting size
      print(
        outpath,
        "{:.2f}MB -> {:.2f}MB".format(size / 1000 ** 2, outsize / 1000 ** 2),
      )
      return os.path.basename(outpath), checksum
    else:
      # if not, replace it with the original file
      os.remove(outpath)
      return cp(path, size, chunk_path)

  def _init_chunk(outdir, prefix, chunk_count):
    chunk_path = os.path.join(outdir, prefix + "{:03}".format(chunk_count + 1))

    if os.path.exists(chunk_path):
      print("fatal: refusing to overwrite", chunk_path, file=sys.stderr)
      sys.exit(1)
    os.mkdir(chunk_path)
    return chunk_path

  cp_fn = gz if gzip else cp

  if shuffle:
    if seed is not None:
      random.seed(seed)
    random.shuffle(paths)

  chunk_count = 0
  init_chunk = lambda: _init_chunk(outdir, prefix, chunk_count)
  chunk_path = init_chunk()

  i = 0
  chunk = []
  chunks = []
  size = 0
  while i < len(paths):
    path, root = paths[i]
    pathsize = os.stat(path).st_size

    if size == 0 and pathsize > maxsize:
      outname, checksum = cp_fn(path, pathsize, chunk_path)
      chunk_meta(
        chunk_path,
        [(outname, os.path.relpath(path, root), pathsize, checksum)],
        pathsize,
        maxsize,
      )
      chunks.append((chunk_path, pathsize))
      chunk_count += 1
      chunk_path = init_chunk()

      i += 1
    elif size + pathsize <= maxsize:
      outname, checksum = cp_fn(path, pathsize, chunk_path)
      chunk.append((outname, os.path.relpath(path, root), pathsize, checksum))
      i += 1
      size += pathsize
    else:
      chunk_meta(chunk_path, chunk, size, maxsize)
      chunks.append((chunk_path, size))
      size = 0
      chunk = []

      chunk_count += 1
      chunk_path = init_chunk()

  # spit out any leftovers
  chunk_meta(chunk_path, chunk, size, maxsize)
  chunks.append((chunk_path, size))

  return chunks


def PathTuplesToChunk(src_dirs: typing.List[pathlib.Path]):
  """Enumerate files in a source directories to chunk.

  Args:
    src_dirs: A list of source directories.

  Returns:
    A list of absolute paths.

  Raises:
    ValueError
  """
  files = []
  for directory in src_dirs:
    if not directory.is_dir():
      raise ValueError(f"{directory} not found")

    for dirpath, _, filenames in os.walk(directory):
      files += [
        (os.path.join(dirpath, filename), dirpath)
        for filename in filenames
        if not filename.startswith("._") and not filename == ".DS_Store"
      ]
  return files


def MakeChunks(
  src_dirs: typing.List[pathlib.Path],
  chunks_dir: pathlib.Path,
  chunk_size_in_bytes: int,
  **kwargs,
) -> None:
  """Create chunk directories.

  Args:
    src_dirs: A list of source directories to chunk.
    chunks_dir: The root path to create chunk directories in.
    chunk_size_in_bytes: The maximum size of each chunk.
    kwargs: Additional arguments to ChunkFiles().
  """
  try:
    files = PathTuplesToChunk(src_dirs)
  except ValueError as e:
    print(f"fatal: {e}", file=sys.stderr)
    sys.exit(1)

  chunks = chunk_files(files, chunks_dir, chunk_size_in_bytes, **kwargs)

  chunksizes_mb = [chunk[1] / 1000 ** 2 for chunk in chunks]
  totalsize_mb = sum(chunksizes_mb)
  avgchunksize_mb = totalsize_mb / len(chunks)
  minchunksize_mb, maxchunksize_mb = min(chunksizes_mb), max(chunksizes_mb)
  num_chunks = len(chunks)

  print(
    f"{num_chunks} chunks of avg size {avgchunksize_mb:.2f} MB "
    f"(min: {minchunksize_mb:.2f} MB, max: {maxchunksize_mb:.2f} MB)"
  )
  print(f"total size of chunks {totalsize_mb:.2f} MB")


def unchunk_file(chunk_path, outdir, manifest_entry, lineno):
  def deflate(src, dst):
    print(src, "->", dst)
    with gzip.open(src, "rb") as infile:
      with open(dst, "wb") as outfile:
        copyfileobj(infile, outfile)

  def cp(src, dst):
    print(src, "->", dst)
    copy(src, dst)

  inpath, checksum, size, outpath = manifest_entry.split("\t")

  inpath = os.path.join(chunk_path, inpath)
  outpath = os.path.join(outdir, outpath)

  if not os.path.exists(inpath):
    raise Exception("file does not exist {inpath}".format(**vars()))
  if os.path.exists(outpath):
    raise Exception("refusing to overwrite file {outpath}".format(**vars()))

  # make parent directories for destination file
  os.makedirs(os.path.dirname(outpath), exist_ok=True)

  ext = os.path.splitext(inpath)[1]  # file extension

  # determine whether file is compressed
  unpack_fn = deflate if ext == ".gz" else cp

  # if file is compressed, unpack it
  unpack_fn(inpath, outpath)

  # validate file size
  try:
    size = int(size)
    actualsize = os.stat(outpath).st_size
    if size != actualsize:
      print(
        "warning[{lineno}]: expected file size {size} does not match"
        "actual size {actualsize}. File is corrupt",
        outpath,
        file=sys.stderr,
      )
  except ValueError:
    print(
      "warning[{lineno}]: could not read file size in manifest".format(
        **vars()
      ),
      file=sys.stderr,
    )

  # validate checksum
  actualchecksum = md5sum(outpath)
  if checksum != actualchecksum:
    print(
      "warning[{lineno}]: checksum validation failed. File is corrupt",
      outpath,
      file=sys.stderr,
    )


def read_manifest(manifestpath):
  # read manifest file
  try:
    with open(manifestpath) as infile:
      return [l for l in infile.read().split("\n") if l]
  except Exception:
    print("fatal: unable to read manifest file", manifestpath)


def unchunk_chunk(chunk_path, out_path):
  chunk = read_manifest(os.path.join(chunk_path, "MANIFEST.txt"))

  for i, row in enumerate(chunk):
    try:
      lineno = i + 1
      unchunk_file(chunk_path, out_path, row, lineno)
    except Exception as e:
      print("error[{lineno}]: {e}".format(**vars()), file=sys.stderr)


def unchunk(chunks_dir: pathlib.Path, out_dir: pathlib.Path):
  chunk_dirs = [d.absolute() for d in chunks_dir.iterdir() if d.is_dir()]
  for directory in chunk_dirs:
    unchunk_chunk(directory, out_dir)
