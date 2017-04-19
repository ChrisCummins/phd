#!/usr/bin/env bash
#
# dotfiles installation script
#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
usage() {
    echo "usage: $0 [-v|--verbose]"
}


version() {
    echo "dotfiles $(git rev-parse --short HEAD)"
}


# path to this repo
dotfiles="$HOME/.dotfiles"
# path to private files
private="$HOME/Dropbox/Shared"


echo_ok() {
    local msg="$@"
    echo -e "$(tput bold)$@$(tput sgr0)"
}


echo_error() {
    local msg="$@"
    echo -e "$(tput bold)$(tput setaf 1)$@$(tput sgr0)" >&2
}


symlink() {
    # symlink a file
    #
    # args:
    #   $1 path that symlink resolves to
    #   $2 fully qualified destination symlink (not just the parent directory)
    #   $3 (optional) prefix. set to "sudo" to symlink with sudo permissions
    local source="$1"
    local destination="$2"
    set +u
    local prefix="$3"
    set -u

    # check that source exists
    if [[ "$source" == /* ]]; then
        local source_abs="$source"
    else
        local source_abs="$(dirname $destination)/$source"
    fi

    if ! $prefix test -f "$source_abs" ; then
        echo_error "failed: symlink $source -> $destination"
        echo_error "error:  symlink source '$source_abs' does not exist"
        exit 1
    fi

    # check that destination is not a file
    if $prefix test -d "$destination" ; then
        echo_error "failed: $source -> $destination"
        echo_error "error:  symlink destination '$destination' is a directory"
        exit 1
    fi

    if ! $prefix test -L "$destination" ; then
        if $prefix test -f "$destination" ; then
            local backup="$destination.backup"
            echo_ok "backup $destination -> $destination.backup"
            $prefix mv "$destination" "$destination.backup"
        fi
        echo_ok "symlink $source -> $destination"
        $prefix ln -s "$source" "$destination"
    fi
}


symlink_dir() {
    # symlink a directory
    #
    # args:
    #   $1 path that symlink resolves to
    #   $2 fully qualified destination symlink (not just the parent directory)
    #   $3 (optional) prefix. set to "sudo" to symlink with sudo permissions
    local source="$1"
    local destination="$2"
    set +u
    local prefix="$3"
    set -u

    # check that source exists
    if [[ "$source" == /* ]]; then
        local source_abs="$source"
    else
        local source_abs="$(dirname $destination)/$source"
    fi

    if ! $prefix test -d "$source_abs" ; then
        echo_error "failed: symlink_dir $source -> $destination"
        echo_error "error:  symlink_dir source '$source_abs' does not exist"
        exit 1
    fi

    # check that destination is not a file
    if $prefix test -f "$destination" ; then
        echo_error "failed: symlink_dir $source -> $destination"
        echo_error "error:  symlink_dir destination '$destination' is a file"
        exit 1
    fi

    if ! $prefix test -L "$destination" ; then
        if $prefix test -d "$destination" ; then
            local backup="$destination.backup"
            echo_ok "backup $destination -> $destination.backup"
            $prefix mv "$destination" "$destination.backup"
        fi
        echo_ok "symlink_dir $source -> $destination"
        $prefix ln -s "$source" "$destination"
    fi
}


clone_git_repo() {
    local url="$1"
    local destination="$2"
    local version="$3"

    if [[ ! -d "$destination" ]]; then
        echo_ok "cloning repo $url to $destination"
        git clone --recursive "$url" "$destination"
    fi

    if [[ ! -d "$destination/.git" ]]; then
        echo_error "failed: cloning repo $url to $destination" >&2
        echo_error "error:  $destination/.git does not exist" >&2
        exit 1
    fi

    cd "$destination"
    local target_hash="$(git rev-parse $version 2>/dev/null)"
    local current_hash="$(git rev-parse HEAD)"
    if [[ "$current_hash" != "$target_hash" ]]; then
        echo_ok "setting repo version $destination to $version"
        git fetch --all
        git reset --hard "$version"
    fi
    cd - &>/dev/null
}


_pip_install() {
    local package="$1"
    local version="$2"

    # on linux, we need sudo to pip install.
    local use_sudo=""
    if [[ "$(uname)" != "Darwin" ]]; then
        use_sudo="sudo"
    fi

    pip freeze 2>/dev/null | grep "^$package==$version" &>/dev/null \
        || $use_sudo pip install --upgrade "$package==$version" 2>/dev/null
}


_npm_install() {
    local package="$1"
    local version="$2"

    npm list -g | grep "$package@$version" &>/dev/null || sudo npm install -g "$package@$version"
}


_apt_get_install() {
    local package="$1"

    dpkg -s "$package" &>/dev/null || sudo apt-get install -y "$package"
}


install_zsh() {
    # install config files
    symlink_dir "$dotfiles/zsh" ~/.zsh
    symlink .zsh/zshrc ~/.zshrc
    if [[ -d "$private/zsh" ]]; then
        symlink_dir "$private/zsh" ~/.zsh/private
    fi

    # oh-my-zsh
    clone_git_repo \
        git@github.com:robbyrussell/oh-my-zsh.git \
        ~/.oh-my-zsh \
        66bae5a5deb7a053adfb05b38a93fe47295841eb

    # syntax highlighting
    clone_git_repo \
        git@github.com:zsh-users/zsh-syntax-highlighting.git \
        ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting \
        ad522a091429ba180c930f84b2a023b40de4dbcc

    # oh-my-zsh config
    symlink ~/.zsh/cec.zsh-theme ~/.oh-my-zsh/custom/cec.zsh-theme

    _pip_install autoenv 1.0.0
}


install_lmk() {
    _pip_install lmk 0.0.3
    symlink "$private/lmk/lmkconfig" ~/.lmk.cfg
}


install_ssh() {
    # shared SSH config
    if [[ -d "$private/ssh" ]]; then
        chmod 600 "$private"/ssh/*
        mkdir -p ~/.ssh
        symlink "$private/ssh/authorized_keys" ~/.ssh/authorized_keys
        symlink "$private/ssh/config" ~/.ssh/config
        symlink "$private/ssh/known_hosts" ~/.ssh/known_hosts
        cp "$private/ssh/id_rsa" ~/.ssh/id_rsa
        symlink "$private/ssh/id_rsa.ppk" ~/.ssh/id_rsa.ppk
        symlink "$private/ssh/id_rsa.pub" ~/.ssh/id_rsa.pub
    fi
}


install_dropbox() {
    mkdir -p ~/.local/bin
    symlink "$dotfiles/dropbox/dropbox.py" ~/.local/bin/dropbox

    if [[ -d ~/Dropbos/Inbox ]]; then
        symlink Dropbox/Inbox ~/Inbox
    fi
}


install_git() {
    # diff-so-fancy requires node stack
    if [[ "$(uname)" == "Darwin" ]]; then
        brew list | grep '^node$' &>/dev/null || brew install npm nodejs
    else
        _apt_get_install nodejs
        _apt_get_install npm
    fi

    _npm_install diff-so-fancy 0.11.4
    symlink .dotfiles/git/gitconfig ~/.gitconfig
}


install_tmux() {
    symlink .dotfiles/tmux/tmux.conf ~/.tmux.conf
}


install_atom() {
    # python linter
    _pip_install pylint 1.7.1
    symlink_dir .dotfiles/atom ~/.atom
    symlink "$dotfiles/atom/ratom" ~/.local/bin/ratom
}


install_vim() {
    symlink "$dotfiles/vim/vimrc" ~/.vimrc

    # Vundle
    clone_git_repo \
        git@github.com:VundleVim/Vundle.vim.git \
        ~/.vim/bundle/Vundle.vim \
        6497e37694cd2134ccc3e2526818447ee8f20f92
    vim +PluginInstall +qall
}


install_sublime() {
    # rsub
    sudo ln -sf "$dotfiles/subl/rsub" /usr/local/bin

    # sublime conf
    if [[ -d "$private/subl" ]] && \
       [[ -d "$HOME/Library/Application Support/Sublime Text 3" ]]; then
        symlink_dir "Library/Application Support/Sublime Text 3" ~/.subl
        symlink_dir "$private/subl/User" ~/.subl/Packages/User
        symlink_dir "$private/subl/INI" ~/.subl/Packages/INI

        # subl
        symlink "/Applications/Sublime Text.app/Contents/SharedSupport/bin/subl" /usr/local/bin/subl sudo
    fi
}

install_ssmtp() {
    if [[ "$(uname)" != "Darwin" ]]; then
        _apt_get_install ssmtp
        symlink "$private/ssmtp/ssmtp.conf" /etc/ssmtp/ssmtp.conf sudo
    fi
}

install_python() {
    # modern python
    if [[ "$(uname)" == "Darwin" ]]; then
        brew list | grep '^python$' &>/dev/null || brew install python
    else
        _apt_get_install python-pip
    fi

    if [[ -f "$private/python/.pypirc" ]]; then
        symlink "$private/python/.pypirc" ~/.pypirc
    fi
}


install_mysql() {
    if [[ -f "$private/mysql/.my.cnf" ]]; then
        symlink "$private/mysql/.my.cnf" ~/.my.cnf
    fi
}


install_tex() {
    if $(which pdflatex &>/dev/null); then
        mkdir -p ~/.local/bin
        symlink "$dotfiles/tex/autotex" ~/.local/bin/autotex
        symlink "$dotfiles/tex/cleanbib" ~/.local/bin/cleanbib
    fi
}


install_omnifocus() {
    # add to OmniFocus cli
    sudo ln -sf "$dotfiles/omnifocus/omni" /usr/local/bin
}


install_server_scripts() {
    case "$(hostname)" in
      florence | diana | mary | plod)
        # server scripts
        symlink "$dotfiles/servers/mary" ~/.local/bin/mary
        symlink "$dotfiles/servers/diana" ~/.local/bin/diana
        ;;
    esac
}


install_macos() {
    # install Mac OS X specific stuff
    mkdir -p ~/.local/bin
    symlink "$dotfiles/macos/rm-dsstore" ~/.local/bin/rm-dsstore

    if [[ "$(uname)" == "Darwin" ]] && [[ -d "$private/macos" ]]; then
        brew list > "$private/macos/brew-$(hostname).txt"
        brew cask list > "$private/macos/brew-$(hostname)-casks.txt"
    fi
}


parse_args() {
    set -e
    if [[ "$1" == "-v" ]] || [[ "$1" == "--verbose" ]]; then
        set -x
        shift
    elif [[ "$1" == "--version" ]]; then
        version
        exit 0
    elif [[ -n "$1" ]]; then
        usage >&2
        exit 1
    fi
    set -u
}


main() {
    parse_args $@
    version

    install_dropbox
    install_ssh
    install_python
    install_zsh
    install_lmk
    install_git
    install_tmux
    install_atom
    install_vim
    install_sublime
    install_ssmtp
    install_mysql
    install_omnifocus
    install_server_scripts
    install_tex
    install_macos
}
main $@
