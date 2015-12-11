#!/usr/bin/env bash
#
# autotex - Automagic LaTeX document management
#
# TODO:
#
#  * Don't update depfile until build has successfully completed.
#  * If no depfile exists, then assume rebuild, then recalculate after
#    compilation.
#  * If the user deletes a depfile, stat should silently error.
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

    if [[ -f $abspath/$document.fls ]]; then
        egrep '^INPUT ' < $1/$2.fls \
            | egrep -v '\.(aux|out|toc|lof|lot|bbl|run\.xml)$' \
            | awk ' !x[$$0]++' \
            | sed 's/^INPUT //' \
            | sed -r 's,^(\./|([^/])),'"$1/\2,"
    else
        find $abspath -name '*.tex'
    fi

    # echo "/Users/cec/src/msc-thesis/docs/thesis/thesis.tex"
}

# Accepts a list of filenames, and prints file modification times and
# names.
#
stat_filenames() {
    xargs $STAT -c '%Y %n'
}

# Determine if the document needs rebuilding.
#
# $1 (str) Absolute path to document directory
# $2 (str) Name of document, without filename extension
#
# FIXME: HAS SIDE EFFECTS, UPDATES DEPFILE.
rebuild_required() {
    local abspath=$1
    local document=$2

    # Make a temporary dependencies file
    scratch=$($MAKETEMP)
    get_dependency_list $abspath $document | stat_filenames > $scratch

    if [[ -f $abspath/$DEPFILE ]]; then
        set +e
        diff $scratch $abspath/$DEPFILE &>/dev/null
        diff=$?
        set -e

        if ! (( $diff )); then
            # Not different: remove the temporary file.
            rm $scratch
            echo 0
            return
        fi
    fi

    # Different: update the depfile.
    mv $scratch $abspath/$DEPFILE
    echo 1
    return
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

    cd - &>/dev/null
    exit
}

main() {
    local command=$1
    local input=$2
    local document=$(basename $input)
    local abspath=$($READLINK -f $(dirname $input))

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
}
main $@
