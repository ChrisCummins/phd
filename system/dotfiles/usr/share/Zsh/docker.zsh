# Aliases and configuration for Docker.

alias d='docker'
alias dp='docker push'
alias dls='docker image ls'
alias drm='docker rmi'
alias dpr='docker system prune'

dtp() {
  docker tag $1 $2;
  docker push $2;
}

# Usage: <bazel_target> <docker_label>
docker_bazel_push() {
  bazel build //$1.tar;
  docker load -i bazel-bin/$1.tar;
  docker tag bazel/$1 $2;
  docker push $2;
}
