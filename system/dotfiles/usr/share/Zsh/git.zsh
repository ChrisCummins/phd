# Git configuration

# No arguments: `git status`. With arguments: acts like `git`
# Extended from: https://github.com/thoughtbot/dotfiles
unalias g
g() {
    GIT=/usr/local/bin/git
    test -f /usr/bin/git && GIT=/usr/bin/git

    if [ -n "$1" ]; then
        $GIT "$@"
    else
        $GIT status
    fi
}

alias ga='git add'
alias gap='git add -p'
alias gc='git commit -v'
alias gca='git commit --amend'
alias gd='git diff'
alias gdc='git diff --cached'
alias gf='git flow'
alias gfr='git pull --rebase'
alias gl='git log'
alias glp='git log -p'
alias gp='git pull --rebase && git push'
alias gz='git fetch'
alias gza='git fetch --all'

# "git + hub = github" wrapper.
[ -x /usr/local/bin/hub ] && alias git='hub'

# clone
clone-me() {
        local repo="$1"
        shift

        git clone git@github.com:ChrisCummins/$repo.git $@
}
