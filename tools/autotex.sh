#!/usr/bin/env bash
#
# autotex - Automagic LaTeX document management
#
# TODO:
#
#  * Add support for "wc", "lint", "cite".
#  * Add a verbose mode which prints to stderr.
#
set -eu

#
# Configurable
#

# GNU coreutils:
MAKETEMP=/usr/local/opt/coreutils/libexec/gnubin/mktemp
READLINK=/usr/local/opt/coreutils/libexec/gnubin/readlink
STAT=/usr/local/opt/coreutils/libexec/gnubin/stat

# LaTeX tools:
PDFLATEX=pdflatex
BIBER=biber
PDFLATEX_ARGS="-recorder -output-format pdf -progname pdflatex -file-line-error -interaction=nonstopmode --shell-escape"

# Filename locations
DEPFILE=.autotex.deps
LOGFILE=.autotex.log
HOOKS_DIRECTORY=scripts

# Return whether a dependency list can be computed.
#
# Parameters:
#
#      $1 Absolute path to directory containing tex source file
#      $2 Basename of tex source file, without the file extension
#
can_compute_dependency_list() {
    local abspath=$1
    local document=$2

    if [[ -f $abspath/$document.fls ]]; then
        echo 1
    else
        echo 0
    fi
}

# Print absolute paths of texfile dependencies
#
# Look first for a '.fls' file. If not found, simply print all of the
# .tex files in the directory, including subdirectories.
#
# Parameters:
#
#      $1 Absolute path to directory containing tex source file
#      $2 Basename of tex source file, without the file extension
#
# Example:
#
#      get_dependency_list /Users/cec document
#      /usr/local/texlive/2015/texmf.cnf
#      /usr/local/texlive/2015/texmf-dist/web2c/texmf.cnf
#      /usr/local/texlive/2015/texmf-var/web2c/pdftex/pdflatex.fmt
#      /Users/cec/document.tex
#      ...
#
get_dependency_list() {
    local abspath=$1
    local document=$2

    # TODO: The list of dependencies should include the script itself:
    # $READLINK -f $0

    echo $abspath/$document.pdf

    # TODO: Add *.pre and *.post executable hooks to dependency list:

    # Scour for dependencies in the .fls file:
    egrep '^INPUT ' < $abspath/$document.fls \
        | egrep -v '\.(aux|out|toc|lof|lot|bbl|run\.xml)$' \
        | awk ' !x[$$0]++' \
        | sed 's/^INPUT //' \
        | sed -r 's,^(\./|([^/])),'"$1/\2,"
}

# Accepts a list of filenames, and prints file modification times and
# names.
#
stat_filenames() {
    xargs $STAT -c '%Y %n' 2>/dev/null
}

# Update dependency file
#
# $1 (str) Absolute path to document directory
# $2 (str) Name of document, without filename extension
# $3 (str) Path to dependency file
#
write_depfile() {
    local abspath=$1
    local document=$2
    local depfile=$3

    get_dependency_list $abspath $document | stat_filenames > $depfile
}

# Determine if the document needs rebuilding.
#
# $1 (str) Absolute path to document directory
# $2 (str) Name of document, without filename extension
rebuild_required() {
    local abspath=$1
    local document=$2

    if (( $(can_compute_dependency_list $abspath $document) )); then

        if [[ -f $abspath/$DEPFILE ]]; then

            # Compute dependency list
            scratch=$($MAKETEMP)
            write_depfile $abspath $document $scratch

            set +e
            diff $scratch $abspath/$DEPFILE &>/dev/null
            diff=$?
            set -e

            rm -f $scratch

            if (( $diff )); then
                echo 1
                return
            else
                echo 0
                return
            fi
        else
            echo 1
        fi

    else
        # Can't compute dependency list, rebuild required.
        echo 1
    fi
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
            echo "  HOOK     $PWD/$hook"
            silent_unless_fail $LOGFILE $hook
        done
    fi
}

# Run pdflatex on document.
#
# $1 (str) Document name, without extension
run_pdflatex() {
    local document=$1

    echo "  LATEX    $PWD/$document.pdf"
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

# Determine whether document contains citations, using \cite{}.
#
# $1 (str) Document name, without extension
contains_citations() {
    local document=$1

    set +e
    grep 'cite{' $document.aux &>/dev/null
    status=$?
    set -e

    if (( $status )); then
        echo 0
    else
        echo 1
    fi
}

# Run biber command.
#
# $1 (str) Document name, without extension
run_biber() {
    local document=$1

    echo "  BIBER    $PWD/$document"
    silent_unless_fail $LOGFILE $BIBER $document
}

# Build LaTeX document.
#
# $1 (str) Absolute path to document directory
# $2 (str) Document name, without extension
build() {
    local abspath=$1
    local document=$2

    cd $abspath

    setup_env
    exec_hooks ".pre"
    run_pdflatex $document
    if (( $(contains_citations $document) )); then
        # TODO: determine which backend to use (biber,bibtex etc.)
        run_biber $document
        run_pdflatex $document
        # TODO: run bibtool and cleanbib
    fi
    run_pdflatex $document
    exec_hooks ".post"

    # Build successful, Update depfile.
    write_depfile $abspath $document $DEPFILE

    cd - &>/dev/null
    exit
}

main() {
    local command=$1
    local input=$2
    local document=$(basename $input)
    local abspath=$($READLINK -f $(dirname $input))

    # Check that directory exists.
    if ! [[ -d "$abspath" ]]; then
        echo "autotex: Directory '$abspath' not found!" >&2
        exit 1
    fi

    # Check that source file exists.
    if ! [[ -f "$abspath/$document.tex" ]]; then
        echo "autotex: File '$abspath/$document.tex' not found!" >&2
        exit 1
    fi

    cd $abspath
    case $command in
        "make")
            if (( $(rebuild_required $abspath $document) )); then
                build $abspath $document
            fi
            ;;
        *)
            echo "autotex: unrecgonised command '$command'!"
            exit 1
            ;;
    esac
    cd - $>/dev/null
}
main $@
