# Creating Docker images through Bazel

Build and pack:

```sh
$ bazel run //learn/docker/bazel:python_image
$ sudo docker save bazel/learn/docker/bazel:python_image | gzip > foo.tar.gz
```

Unpack and run:

```sh
$ gunzip -c foo.tar.gz | sudo docker load
$ docker run bazel/learn/docker/bazel:python_image
```
