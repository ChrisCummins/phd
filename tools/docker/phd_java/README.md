# Docker image: phd_base_java

An extension to `phd_base` with Java.

To create a docker image from a `py_binary` target, create a `py3_image` target
and set the `base` to `@base//image`, e.g:

```sh
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py3_image(
    name = "image",
    srcs = [":foo"],
    main = ["foo.py"],
    base = "@base_java//image",
    deps = [":foo"],
)
```
