alias g='git'
alias gA='git add'
alias gAp='git add -p'
alias gC='git commit'
alias gCa='git commit --amend'
alias gD='git diff'
alias gDc='git diff --cached'
alias gF='git pull'
alias gFr='git pull --rebase'
alias gL='git log'
alias gLp='git log -p'
alias gP='git push'
alias gS='git status'
alias gZ='git fetch'
alias gZa='git fetch --all'

# "git + hub = github" wrapper.
[ -x /usr/local/bin/hub ] && alias git='hub'

# clone
clone-me() {
        local repo="$1"
        shift

        git clone git@github.com:ChrisCummins/$repo.git $@
}
