# Shutterbug

Shutterbug splits a large collection of photos into a series of DVD-sized
folders for burning to disc, and provides a mechanism to go from these disc
backups back to the original directory structure.

Shutterbug is paranoid about data loss and corruption, and has some coping
strategies:

* It maintains a 1-1 mappings so that 1 input file = 1 file on a disc. This
  means if a file becomes corrupted, you lose one image. Other approaches which
  do not respect file boundaries lead to losing multiple images in one go.
* It randomizes the order of the files on the discs so that if a disc is lost,
  you end up with lots of tiny gaps in your photo library, not one big one.
* It validates your files after restoring from backup, so you have a warning of
  data corruption. Note this does not **prevent** data loss, only **discovers**
  it. *The best way to prevent irrecoverable data loss is to make more copies of
  your data to begin with.*


## Usage


### Archiving to disc

To backup your photo library in `~/Pictures/2016` to 4.7GB DVDs, split the
folder into "chunks" using shutterbug:

```
$ mkdir ~/chunks
$ bazel run //util/photolib/shutterbug:pack \
    --gzip --size=4695 --chunk_prefix=chunk_ \
    --src_dir=$HOME/Pictures/2016 --chunks_dir=$HOME/chunks
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


### Restoring from disc

Copy each chunk from your DVDs back to disc, e.g. `~/import/chunk_001`,
`~/import/chunk_002` etc. Restore the original file structure by running
shutterbug from the output directory:

```
$ mkdir ~/Pictures/2016
$ bazel run //util/photolib/shutterbug:unpack \
    --chunks_dir=$HOME/import --out_dir=$HOME/Pictures/2016
~/import/chunk_001/ae3d47f87af176b74e1ec30599a7b31a.jpg.gz -> ./2016-12 NYC (2434 of 5025).jpg
~/import/chunk_001/631600d1e11339794e81d75f104e9f19.jpg.gz -> ./2016-12 NYC (4411 of 5025).jpg
~/import/chunk_001/130c52fe396237a59500a61b8101ff55.jpg.gz -> ./2016-12 NYC (301 of 5025).jpg
...
```

Shutterbug will print warnings for files in case the size or contents have
changed.
