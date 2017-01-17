# Shutterbug

Shutterbug solves the simple problem that I have a bunch of photos I'd like to
burn to DVDs for safe keeping. It does two things: splits collections of files
into size-limited "chunks", and recovers those chunks back into the original
directory trees. Naturally this turned into an exploration of knapsack problems,
packing, and other NP complete fun.

Shutterbug is paranoid about data loss and corruption, and has some coping
strategies:

* It maintains a 1-1 mappings so that 1 input file = 1 file on a disk.
* It shuffles the order of the files on the disks so that if a disk is lost, you
  end up with lots of tiny gaps in your photo library, not one big one.
* It validates the contents of files before and after copying to disk so that
  you have a warning of data corruption.

## Usage

Install using:

```
$ python ./setup.py install
```

### Archiving

To burn your old photos in `~/Pictures/2016` to 4.7GB DVDs, create the required
"chunks":

```
$ mkdir ~/chunks && cd ~/chunks
$ shutterbug ~/Pictures/2016 --size 4700 --gzip
chunk_001/ae3d47f87af176b74e1ec30599a7b31a.jpg.gz 4.93MB -> 4.90MB
chunk_001/631600d1e11339794e81d75f104e9f19.jpg.gz 7.40MB -> 7.38MB
chunk_001/130c52fe396237a59500a61b8101ff55.jpg.gz 6.79MB -> 6.77MB
chunk_001/27fc10914e18b0e1b303c05a800c299d.jpg.gz 5.73MB -> 5.70MB
...
Wrote chunk_001/MANIFEST.txt
Wrote chunk_001/README.txt
chunk_001 has 723 files, size 4662.40 MB (99.2% of maximum size)

chunk_002/5c9ce3b8071207ab702766ac2be76f10.jpg.gz 6.13MB -> 6.11MB
chunk_002/bc17480a318e7ba9a3e4e2e57538917d.jpg.gz 9.63MB -> 9.60MB
...
```

Burn each of the resulting folders in `~/chunks` to DVDs.

### Unarchiving

Copy each chunk from your DVDs back to disk, e.g. `~/import/chunk_001`,
`~/import/chunk_002` etc. Restore original file structure by running shutterbug
from the output directory:

```
$ mkdir ~/Pictures/2016 && cd ~/Pictures/2016
$ shutterbug --unpack ~/import/chunk_*
$ ~/Pictures/mkbackup.py ../compressed/* -u
~/import/chunk_001/ae3d47f87af176b74e1ec30599a7b31a.jpg.gz -> ./2016-12 NYC (2434 of 5025).jpg
~/import/chunk_001/631600d1e11339794e81d75f104e9f19.jpg.gz -> ./2016-12 NYC (4411 of 5025).jpg
~/import/chunk_001/130c52fe396237a59500a61b8101ff55.jpg.gz -> ./2016-12 NYC (301 of 5025).jpg
...
```

Shutterbug will print warnings for files in case the size or contents have
changed.
