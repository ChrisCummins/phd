#!/usr/bin/env bash
#
# autotex - Build a LaTeX document
#
set -eu

usage() {
    echo "Usage: $0 <input-tex> <output-pdf>"
}

#
# Configurable
#

# LaTeX tools:
PDFLATEX=pdflatex
BIBER=biber
BIBTEX=bibtex
PDFLATEX_ARGS="-recorder -output-format pdf -progname pdflatex -file-line-error -interaction=nonstopmode --shell-escape"

# GNU coreutils:
if [ "$(uname)" == "Darwin" ]; then
    MAKETEMP=/usr/local/opt/coreutils/libexec/gnubin/mktemp
    READLINK=/usr/local/opt/coreutils/libexec/gnubin/readlink
    STAT=/usr/local/opt/coreutils/libexec/gnubin/stat
else
    MAKETEMP=mktemp
    READLINK=readlink
    STAT=stat
fi

# Included tools:
ROOT=~/phd

# Filename locations
LOGFILE=.autotex.log
HOOKS_DIRECTORY=scripts

# Output formatting:
#
# Note: tput can return with non-zero status in some TTY environments,
# so we disable errors.
set +e
TTYreset="$(tput sgr0 2>/dev/null)"
TTYbold="$(tput bold 2>/dev/null)"
TTYstandout="$(tput smso 2>/dev/null)"
TTYunderline="$(tput smul 2>/dev/null)"
TTYblack="$(tput setaf 0 2>/dev/null)"
TTYblue="$(tput setaf 4 2>/dev/null)"
TTYcyan="$(tput setaf 6 2>/dev/null)"
TTYgreen="$(tput setaf 2 2>/dev/null)"
TTYmagenta="$(tput setaf 5 2>/dev/null)"
TTYred="$(tput setaf 1 2>/dev/null)"
TTYwhite="$(tput setaf 7 2>/dev/null)"
TTYyellow="$(tput setaf 3 2>/dev/null)"
set -e

TaskNameLength=8

# Print a message.
#
print() {
    local body="$1"
    local format="$2"

    echo "$format$body$TTYreset"
}

# Print a formatted task message.
#
print_task() {
    local name="$1"
    local path="$2"
    local format="$3"

    printf "  $TTYbold$format%-${TaskNameLength}s$TTYreset %s\n" "$name" "$path"
}


# Log output of a command silently. If command fails, print log.
#
# $1 (str) Path to log file
# $@ Command arguments to execute
silent_unless_fail() {
    local log=$1
    shift
    local command=$@

    set +e
    # FIXME: Append logfile instead of overwriting, but still redirect
    # all outputs (not just stdout and stderr):
    $command &> $log
    status=$?
    set -e

    if (( $status )); then
        echo "*** Build failed! Autotex log:" >&2
        echo
        cat $log >&2
        echo
        echo "*** End of autotex log. See: $log" >&2
        exit 1
    fi
}

# Execute all hooks for a given file suffix.
#
# $1 Hook file suffix
exec_hooks() {
    local suffix=$1

    # Check whether we should actually run hooks
    set +u
    if [[ -n "$NO_HOOKS" ]]; then
        set -u
        return
    fi
    set -u

    local hooks_glob=$HOOKS_DIRECTORY/*$suffix
    local hooks=$(ls $hooks_glob 2>/dev/null)

    if [[ -n "$hooks" ]]; then
        for hook in $hooks; do
            print_task "HOOK" "$hook" "$TaskMisc"
            silent_unless_fail $LOGFILE $hook
        done
    fi
}

# Run pdflatex on document.
#
# $1 (str) Document name, without extension
# $2 (str) Format string
run_pdflatex() {
    local document=$1
    local format=$2

    print_task "LATEX" "$document.pdf" "$format"
    silent_unless_fail $LOGFILE $PDFLATEX $PDFLATEX_ARGS $document.tex
}

# Setup environment variables and autotex env.
#
setup_env() {
    # Export environment variables
    export TEXINPUTS=.:./lib:

    # Remove the old logfile.
    rm -f $LOGFILE
}

# Determine which citation backend is used. Returns an empty string if
# none is found.
#
# $1 (str) Document name, without extension
get_citation_backend() {
    local document=$1

    #
    # We determine the citation backend by searching the aux file for
    # different keywords. Note that the order here is important. We
    # must search first for bibtex _before_ biber, since bibtex uses a
    # \bibcite{} keyword, which would test true for the biber backend.
    #
    set +e
    if grep 'citation{' $document.aux &>/dev/null ; then
        echo "bibtex"
    elif grep 'cite{' $document.aux &>/dev/null ; then
        echo "biber"
    else
        echo # No citations.
    fi
    set -e
}

# Run citation backend command.
#
# $1 (str) Backend command name
# $2 (str) Document name, without extension
# $3 (str) Format string
run_citation_backend() {
    local backend=$1
    local document=$2
    local format=$3

    case $backend in
        "biber")
            print_task "BIBER" "$document" "$format"
            silent_unless_fail $LOGFILE $BIBER $document
            ;;
        "bibtex")
            print_task "BIBTEX" "$document" "$format"
            silent_unless_fail $LOGFILE $BIBTEX $document
            ;;
        *)
            echo "autotex: unrecognized backend '$backend'!" >&2
            exit 1
            ;;
    esac
}

main() {
    if [ "$#" -ne 2 ]; then
        usage
        exit 1
    fi

    local input_file="$1"
    local output_file="$2"

    local input_dir=$($READLINK -f $(dirname $input_file))
    local output_dir=$($READLINK -f $(dirname $output_file))

    local input_document="$(basename ${input_file%.tex})"
    local output_document="$(basename ${output_file%.pdf})"


    # temporary directory
    tmpdir="$(mktemp -p $output_dir -d --suffix=.$input_document.autotex)"
    if [[ ! "$tmpdir" || ! -d "$tmpdir" ]]; then
        echo "autotex: failed to create temporary directory" >&2
        exit 2
    fi

    rm_tmpdir() {
        rm -rf "$tmpdir"
    }
    trap rm_tmpdir EXIT

    cd $tmpdir

    # copy the sources into the working directory. When symlinks are
    # encountered, the file they point to is copied.
    rsync -a --copy-links $input_dir/ $tmpdir/

    setup_env

    exec_hooks ".pre"
    run_pdflatex $input_document "$TTYred"

    local backend=$(get_citation_backend $input_document)
    if [[ -n $backend ]]; then
        run_citation_backend $backend $input_document "$TTYblue"
        run_pdflatex $input_document "$TTYyellow"
        # TODO: run bibtool and cleanbib
    fi

    run_pdflatex $input_document "$TTYgreen"
    exec_hooks ".post"

    # copy from temporary directory to output
    cd - &>/dev/null
    mkdir -p "$output_dir"
    mv "$tmpdir/$input_document.pdf" "$output_file"
}
main $@
