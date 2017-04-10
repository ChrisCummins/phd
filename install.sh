#!/usr/bin/env bash
set -e
if [[ "$1" == "-v" ]] || [[ "$1" == "--verbose" ]]; then
    set -x
    shift
fi
set -u


private="$HOME/Dropbox/Shared"


symlink() {
    # args:
    #   $1 path that symlink resolves to
    #   $2 fully qualified destination symlink (not just the parent directory)
    local source="$1"
    local destination="$2"

    if [[ ! -L "$destination" ]]; then
        if [[ -f "$destination" ]]; then
            local backup="$destination.backup"
            echo "backup $destination -> $destination.backup"
            mv "$destination" "$destination.backup"
        fi
        echo "symlink $source -> $destination"
        ln -s "$source" "$destination"
    fi
}


symlink_dir() {
    # args:
    #   $1 path that symlink resolves to
    #   $2 fully qualified destination symlink (not just the parent directory)
    local source="$1"
    local destination="$2"

    if [[ ! -L "$destination" ]]; then
        if [[ -d "$destination" ]]; then
            local backup="$destination.backup"
            echo "backup $destination -> $destination.backup"
            mv "$destination" "$destination.backup"
        fi
        echo "symlink $source -> $destination"
        ln -s "$source" "$destination"
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

    npm list -g | grep "$package" &>/dev/null || sudo npm install -g "$pacakge"
}


install_packages() {
    set +u
    [ -f /etc/os-release ] && source /etc/os-release
    [ -z "$NAME" ] && export NAME=$(uname)
    set -u

    if [[ "$NAME" == "Darwin" ]]; then
        brew list | grep '^node$' &>/dev/null || brew install npm nodejs
        brew list | grep '^python$' &>/dev/null || brew install python
    elif [[ "$NAME" == "Ubuntu" ]]; then
        _apt_get_install nodejs
        _apt_get_install npm
        _apt_get_install python-pip
    fi

    _pip_install autoenv
    _npm_install diff-so-fancy
}


install_zsh() {
    # install config files
    symlink ~/.dotfiles/zsh ~/.zsh
    symlink .zsh/zshrc ~/.zshrc

    # install oh-my-zsh
    clone_repo git@github.com:robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
    symlink ~/.zsh/cec.zsh-theme ~/.oh-my-zsh/custom/cec.zsh-theme
    clone_repo https://github.com/zsh-users/zsh-syntax-highlighting.git ~/oh-my-zsh/custom/plugins/zsh-syntax-highlighting

    # install local config files
    mkdir -p ~/.zsh/private
    if [[ -d "$private/zsh" ]]; then
        symlink "$private/zsh/diana.zsh" ~/.zsh/private/diana.zsh
        symlink "$private/zsh/mary.zsh" ~/.zsh/private/mary.zsh
        symlink "$private/zsh/omni.zsh" ~/.zsh/private/omni.zsh

        # Mac-specific shell stuff
        if [[ "$(uname)" == "Darwin" ]]; then
            symlink "$private/zsh/homebrew.zsh" ~/.zsh/private/homebrew.zsh
        fi
    fi

    clone_repo https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting
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
    symlink .dotfiles/git/gitconfig ~/.gitconfig
}


install_tmux() {
    symlink .dotfiles/tmux/tmux.conf ~/.tmux.conf
}


install_atom() {
    # python linter
    _pip_install pylint
    symlink .dotfiles/atom ~/.atom
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
        symlink "Library/Application Support/Sublime Text 3" ~/.subl
        symlink "$private/subl/User" ~/.subl/Packages/User
        symlink "$private/subl/INI" ~/.subl/Packages/INI

        # subl
        sudo ln -sf \
            "/Applications/Sublime Text.app/Contents/SharedSupport/bin/subl" \
            /usr/local/bin
    fi
}

install_python() {
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
    install_packages
    install_dropbox
    install_ssh
    install_zsh
    install_git
    install_tmux
    install_atom
    install_vim
    install_sublime
    install_python
    install_mysql
    install_omnifocus
    install_server_scripts
    install_tex
    install_macos
}
main $@
