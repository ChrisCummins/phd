# Exporting a pre-trained CLgen model to a Docker image

Usage:

```sh
$ ./make_image.sh
```

TODO:
* Symlinks between CLgen components should use relative paths, so that they
  don't break when exporting.
* Neither the pre-processed or encoded corpuses are necessary for sampling.
  Remove them.
* `Corpus.id` field is not yet implemented, so we have to copy the contentfiles
  into the image.
