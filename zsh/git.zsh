# Git goodness.
alias g='git'
alias gA='git add'
alias gAp='git add -p'
alias gC='git commit'
alias gD='git diff'
alias gF='git pull'
alias gFr='git pull --rebase'
alias gZ='git fetch'
alias gZa='git fetch --all'
alias gL='git log'
alias gP='git push'
alias gS='git status'

# "git + hub = github" wrapper.
[ -x /usr/local/bin/hub ] && alias git='hub'
