Photolib is an opinionated set of rules for structuring libraries of photos,
and a tool to check for errors.

$ photolib init

```
//WORKSPACE
//photos
//third_party
//lightroom
```

```
//photos/2019/2019-08/2019-08-20/20190820T110423.dng
//photos/2019/2019-08/2019-08-20/20190820T110423-2.dng
//photos/2019/2019-08/2019-08-20/20190820T110423-2-HDR.dng
//photos/2019/2019-08/2019-08-20/20190820T110423-2-Edit.dng
```

# cd to directory
$ photolib lint
