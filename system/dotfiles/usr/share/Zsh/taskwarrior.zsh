# Aliases and configuration for task warrior.
# https://taskwarrior.org/

alias t='task'
alias ta="task add"
alias td="task done"
alias t-="task delete"
alias tm="task modify"
alias tu="task undo"
alias ts="task sync"
alias tsh="tasksh"
alias tv='vit rc.alias.next=list'

alias tc="task context"
alias tt="task rc.context=none"

alias tb="task burndown.daily"
alias tbw="task burndown.weekly"
alias tbm="task burndown.monthly"

tp() {
	if [[ -n "$@" ]]; then
		task project:"$@"
	else
		task projects
	fi
}
