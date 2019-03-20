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
