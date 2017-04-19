#!/usr/bin/env bash
#
# dotfiles install script
#
usage() {
    echo "usage: $0 [-v|--verbose]"
}

version() {
    echo "dotfiles $(git rev-parse --short HEAD)"
}

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


private="$HOME/Dropbox/Shared"


symlink() {
    # args:
    #   $1 path that symlink resolves to
    #   $2 fully qualified destination symlink (not just the parent directory)
    #   $3 (optional) prefix. set to "sudo" to symlink with sudo permissions
    local source="$1"
    local destination="$2"
    set +u
    local prefix="$3"
    set -u

    if [[ "$source" == /* ]]; then
        local source_abs="$source"
    else
        local source_abs="$(dirname $destination)/$source"
    fi

    if ! $prefix test -f "$source_abs" ; then
        echo "error: symlink source '$source_abs' does not exist" >&2
        exit 1
    fi

    if $prefix test -d "$destination" ; then
        echo "error: symlink destination '$destination' is a directory" >&2
        exit 1
    fi

    if ! $prefix test -L "$destination" ; then
        if $prefix test -f "$destination" ; then
            local backup="$destination.backup"
            echo "backup $destination -> $destination.backup"
            $prefix mv "$destination" "$destination.backup"
        fi
        echo "symlink $source -> $destination"
        $prefix ln -s "$source" "$destination"
    fi
}


symlink_dir() {
    # args:
    #   $1 path that symlink resolves to
    #   $2 fully qualified destination symlink (not just the parent directory)
    #   $3 (optional) prefix. set to "sudo" to symlink with sudo permissions
    local source="$1"
    local destination="$2"
    set +u
    local prefix="$3"
    set -u

    if [[ "$source" == /* ]]; then
        local source_abs="$source"
    else
        local source_abs="$(dirname $destination)/$source"
    fi

    if ! $prefix test -d "$source_abs" ; then
        echo "error: symlink_dir source '$source_abs' does not exist"
        exit 1
    fi

    if $prefix test -f "$destination" ; then
        echo "error: symlink_dir destination '$destination' is a file" >&2
        exit 1
    fi

    if ! $prefix test -L "$destination" ; then
        if $prefix test -d "$destination" ; then
            local backup="$destination.backup"
            echo "backup $destination -> $destination.backup"
            $prefix mv "$destination" "$destination.backup"
        fi
        echo "symlink_dir $source -> $destination"
        $prefix ln -s "$source" "$destination"
    fi
}


clone_repo() {
    local url="$1"
    local destination="$2"

    if [[ ! -d "$destination" ]]; then
        echo "cloning repo $destination"
        git clone --recursive "$url" "$destination"
    fi
}


_pip_install() {
    local package="$1"

    # on linux, we need sudo to pip install.
    local use_sudo=""
    if [[ "$(uname)" != "Darwin" ]]; then
        use_sudo="sudo"
    fi

    pip freeze 2>/dev/null | grep "^$package" &>/dev/null \
        || $use_sudo pip install "$package" 2>/dev/null
}


_apt_get_install() {
    local package="$1"

    dpkg -s "$package" &>/dev/null || sudo apt-get install -y "$package"
}


_npm_install() {
    local package="$1"

    npm list -g | grep "$package" &>/dev/null || sudo npm install -g "$package"
}


install_zsh() {
    # install config files
    symlink_dir ~/.dotfiles/zsh ~/.zsh
    symlink .zsh/zshrc ~/.zshrc
    if [[ -d "$private/zsh" ]]; then
        symlink_dir "$private/zsh" ~/.zsh/private
    fi

    # install oh-my-zsh
    clone_repo git@github.com:robbyrussell/oh-my-zsh.git ~/.oh-my-zsh

    # syntax highlighting
    clone_repo https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting

    # oh-my-zsh config
    symlink ~/.zsh/cec.zsh-theme ~/.oh-my-zsh/custom/cec.zsh-theme

    _pip_install autoenv
}


install_lmk() {
    _pip_install lmk
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
    symlink ~/.dotfiles/dropbox/dropbox.py ~/.local/bin/dropbox

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

    _npm_install diff-so-fancy
    symlink .dotfiles/git/gitconfig ~/.gitconfig
}


install_tmux() {
    symlink .dotfiles/tmux/tmux.conf ~/.tmux.conf
}


install_atom() {
    # python linter
    _pip_install pylint
    symlink_dir .dotfiles/atom ~/.atom
    symlink ~/.dotfiles/atom/ratom ~/.local/bin/ratom
}


install_vim() {
    symlink .dotfiles/vim/vimrc ~/.vimrc
    clone_repo https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
    vim +PluginInstall +qall
}


install_sublime() {
    # rsub
    sudo ln -sf "$HOME/.dotfiles/subl/rsub" /usr/local/bin

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
        symlink "$HOME/.dotfiles/tex/autotex" ~/.local/bin/autotex
        symlink "$HOME/.dotfiles/tex/cleanbib" ~/.local/bin/cleanbib
    fi
}


install_omnifocus() {
    # add to OmniFocus cli
    sudo ln -sf "$HOME/.dotfiles/omnifocus/omni" /usr/local/bin
}


install_server_scripts() {
    case "$(hostname)" in
      florence | diana | mary | plod)
        # server scripts
        symlink "$HOME/.dotfiles/servers/mary" ~/.local/bin/mary
        symlink "$HOME/.dotfiles/servers/diana" ~/.local/bin/diana
        ;;
    esac
}


install_macos() {
    # install Mac OS X specific stuff
    mkdir -p ~/.local/bin
    symlink "$HOME/.dotfiles/macos/rm-dsstore" ~/.local/bin/rm-dsstore

    if [[ "$(uname)" == "Darwin" ]] && [[ -d "$private/macos" ]]; then
        brew list > "$private/macos/brew-$(hostname).txt"
        brew cask list > "$private/macos/brew-$(hostname)-casks.txt"
    fi
}


main() {
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
