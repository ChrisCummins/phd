# Print some statistics about a git repository.
#
# Usage: (in a git repo)
#
#    $PHD/tools/git/repo_stats.sh
set -eu

stats() {
  local dir="$1"

  num_contributors=$(git shortlog -s -n | awk '{$1=""}1' | sort -u | wc -l)
  # Wage estimate using 2019 USA software developer salary from Glassdoor.
  sloccount --personcost 103035 . 2>&1 > .git_stats_sloc.txt
  loc=$(grep 'Total Physical Source Lines of Code (SLOC)' .git_stats_sloc.txt | cut -d'=' -f2)
  cost=$(grep 'Total Estimated Cost to Develop' .git_stats_sloc.txt | cut -d'=' -f2)
  first_commit_year=$(git show -s --format=%ci $(git rev-list HEAD | tail -n 1) | cut -d'-' -f1)
  remote=$(git remote -v | head -n1 | cut -d' ' -f1 | cut -d$'\t' -f2 | sed 's/\.git$//')

  echo "Remote:                     $remote"
  echo "Development started:        $first_commit_year"
  echo "Num contributors:           $num_contributors"
  echo "Logical lines of code:     $loc"
  echo "Estimated cost to develop: $cost"

  rm .git_stats_sloc.txt
}

stats $(pwd)
