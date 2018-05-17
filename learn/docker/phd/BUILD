# This package contains defines a docker container for this repo.

genrule(
    name = "phd_image",
    srcs = ["Dockerfile"],
    outs = ["phd_docker_image.tar"],
    cmd = "cp -v $< $(@D) && sudo docker build -t phd $(@D) && sudo docker save phd -o $@",
)
