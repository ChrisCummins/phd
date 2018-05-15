Build the docker image using:

```
$ sudo docker build -t phd_image
```

Run the docker image as a container:

```
$ sudo docker run -it phd_image /bin/zsh
```

On MacOS you don't need to run the docker commands with `sudo` since it uses
a VM.
