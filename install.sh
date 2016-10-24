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
    #   $2 fully qualified destination symlink (not just the directory name)
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


clone_repo() {
    local url="$1"
    local destination="$2"

    if [[ ! -d "$destination" ]]; then
        echo "cloning repo $destination"
        git clone --recursive "$url" "$destination"
    fi
}


install_zsh() {
    # Install zsh configuration
    clone_repo git@github.com:ChrisCummins/zsh.git ~/.local/src/zsh
    symlink .local/src/zsh/zshrc ~/.zshrc

    # symlink for config directories
    symlink .local/src/zsh ~/.zsh
    symlink .local/src/zsh/oh-my-zsh ~/.oh-my-zsh

    cd ~/.zsh
    echo "Updating ~/.zsh"
    git pull origin master
    git submodule update

    # install local config files
    if [[ -d "$private/zsh" ]]; then
        symlink "$private/zsh/diana.zsh" ~/.zsh/local/diana.zsh
        symlink "$private/zsh/git.zsh" ~/.zsh/local/git.zsh
        symlink "$private/zsh/mary.zsh" ~/.zsh/local/mary.zsh
        symlink "$private/zsh/omni.zsh" ~/.zsh/local/omni.zsh

        # Mac-specific shell stuff
        if [[ "$(uname)" == "Darwin" ]]; then
            symlink "$private/zsh/homebrew.zsh" ~/.zsh/local/homebrew.zsh
        fi
    fi
}


install_ssh() {
    if [[ -d "$private/ssh" ]]; then
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
    symlink ~/.dotfiles/dropbox/dropbox.py ~/.local/bin/dropbox
}


install_git() {
    symlink .dotfiles/git/gitconfig ~/.gitconfig
}


install_tmux() {
    symlink .dotfiles/tmux/tmux.conf ~/.tmux.conf
}


install_vim() {
    symlink .dotfiles/vim/vimrc ~/.vimrc
    clone_repo https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
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


install_inbox() {
    if [[ -d ~/Dropbos/Inbox ]]; then
        symlink Dropbox/Inbox ~/Inbox
    fi
}


install_tex() {
    if $(which pdflatex &>/dev/null); then
        mkdir -p ~/.local/bin
        symlink "$HOME/.dotfiles/tex/autotex" ~/.local/bin/autotex
        symlink "$HOME/.dotfiles/tex/cleanbib" ~/.local/bin/cleanbib
    fi
}


install_omni() {
    # add to OmniFocus cli
    sudo ln -sf "$HOME/.dotfiles/omni/omni" /usr/local/bin
}


install_servers() {
    # server scripts
    case "$(hostname)" in
    florence | diana | mary)
        symlink "$HOME/.dotfiles/servers/mary" ~/.local/bin/mary
        symlink "$HOME/.dotfiles/servers/diana" ~/.local/bin/diana
        ;;
    esac
}


main() {
    install_ssh
    install_zsh
    install_dropbox
    install_git
    install_tmux
    install_vim
    install_sublime
    install_inbox
    install_tex
    install_omni
    install_servers
}
main $@
