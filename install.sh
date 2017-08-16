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

hostname="$(hostname)"

echo_ok() {
    local msg="$@"
    echo -e "$(tput bold)$@$(tput sgr0)"
}


echo_error() {
    local msg="$@"
    echo -e "$(tput bold)$(tput setaf 1)$@$(tput sgr0)" >&2
}


copy() {
    # copy a file
    #
    # args:
    #   $1 copy source
    #   $2 fully qualified copy destination (not just the parent directory)
    local source="$1"
    local destination="$2"

    if ! test -f "$source" ; then
        echo_error "failed: cp $source -> $destination"
        echo_error "error:  source $source does not exist"
        exit 1
    fi

    if ! diff "$source" "$destination" &>/dev/null ; then
        echo_ok "copy $source -> $destination"
        cp "$source" "$destination"
    fi
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
        echo_ok "cloning $url -> $destination"
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


_brew_list_cache="$dotfiles/.brew-list.txt"
_npm_list_cache="$dotfiles/.npm-list.txt"
_brew_cask_list_cache="$dotfiles/.brew-cask-list.txt"
_cache_cleanup() {
    rm $dotfiles/.*-freeze.txt
    rm -f \
        "$_npm_list_cache" \
        "$_brew_list_cache" \
        "$_brew_cask_list_cache"
}
trap _cache_cleanup EXIT


pip_freeze() {
    local pip="${1:-pip}"
    local use_sudo="${2:-}"

    # on linux, we need sudo to pip install.
    if [[ "$(uname)" != "Darwin" ]]; then
        use_sudo="sudo -H"
    fi

    local pip_freeze_cache="$dotfiles/.$pip-freeze.txt"
    if [[ ! -f "$pip_freeze_cache" ]] ; then
        $use_sudo $pip freeze 2>/dev/null > "$pip_freeze_cache"
    fi
    echo "$pip_freeze_cache"
}


_pip_install() {
    local package="$1"
    local version="$2"
    local pip="${3:-pip}"
    local use_sudo="${4:-}"

    # on linux, we need sudo to pip install.
    if [[ "$(uname)" != "Darwin" ]]; then
        use_sudo="sudo -H"
    fi

    grep "^$package==$version" < "$(pip_freeze "$pip")" >/dev/null || { \
        echo_ok "$pip install $package==$version"; \
        $use_sudo $pip install --upgrade "$package==$version" >/dev/null; \
    }
}


npm_list() {
    if [[ ! -f "$_npm_list_cache" ]] ; then
        npm list -g > "$_npm_list_cache"
    fi
    echo "$_npm_list_cache"
}


_npm_install() {
    local package="$1"
    local version="$2"

    grep "$package@$version" < "$(npm_list)" >/dev/null || \
        sudo npm install -g "$package@$version"
}


brew_list() {
    if [[ ! -f "$_brew_list_cache" ]] ; then
        brew list > "$_brew_list_cache"
    fi

    echo "$_brew_list_cache"
}


brew_install() {
    local package="$1"
    # warning: homebrew packages are not verioned

    if ! grep "^$package$" < "$(brew_list)" >/dev/null; then
        echo_ok "brew install $package"
        brew install "$package"
    fi
}


brew_cask_list() {
    if [[ ! -f "$_brew_cask_list_cache" ]] ; then
        brew cask list > "$_brew_cask_list_cache"
    fi
    echo "$_brew_cask_list_cache"
}


brew_cask_install() {
    local package="$1"
    # warning: homebrew casks are not versioned

    if ! grep "^$package$" < "$(brew_cask_list)" >/dev/null; then
        echo_ok "brew cask install $pacakge"
        brew cask install "$package"
    fi
}


_apt_get_install() {
    local package="$1"
    # warning: debian packages are not verioned

    if ! dpkg -s "$package" &>/dev/null ; then
        echo_ok "apt-get install $Package"
        sudo apt-get install -y "$package"
    fi
}


# Global state:
installed_netdata=0


install_homebrew() {
    if [[ "$(uname)" == "Darwin" ]]; then
        if ! which brew >/dev/null ; then
            echo_ok "installing homebrew"
            /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
            brew update
            brew doctor
        fi

        if ! which brew-cask-outdated >/dev/null ; then
            echo_ok "installing brew-cask-outdated"
            local version="2f08b5a76605fbfa9ab0caeb4868a29ef7e69bb1"
            curl https://raw.githubusercontent.com/bgandon/brew-cask-outdated/$version/brew-cask-outdated.sh 2>/dev/null > ~/.local/bin/brew-cask-outdated
            chmod +x ~/.local/bin/brew-cask-outdated
        fi
    fi
}


install_zsh() {
    if [[ "$(uname)" == "Darwin" ]]; then
        brew_install zsh
    fi

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
    _pip_install lmk 0.0.13
    if [[ -d "$private/lmk" ]]; then
        symlink "$private/lmk/lmkrc" ~/.lmkrc
    fi
}


install_ssh() {
    if [[ -d "$private/ssh" ]]; then
        chmod 600 "$private"/ssh/*
        mkdir -p ~/.ssh

        if [[ "$(stat -c %U "$private/ssh/authorized_keys")" == "$USER" ]]; then
            symlink "$private/ssh/authorized_keys" ~/.ssh/authorized_keys
        else
            copy "$private/ssh/authorized_keys" ~/.ssh/authorized_keys
        fi

        if [[ "$(stat -c %U "$private/ssh/authorized_keys")" == "$USER" ]]; then
            symlink "$private/ssh/config" ~/.ssh/config
        else
            copy "$private/ssh/config" ~/.ssh/config
        fi

        if [[ "$(stat -c %U "$private/ssh/authorized_keys")" == "$USER" ]]; then
            symlink "$private/ssh/known_hosts" ~/.ssh/known_hosts
        else
            copy "$private/ssh/known_hosts" ~/.ssh/known_hosts
        fi

        copy "$private/ssh/id_rsa" ~/.ssh/id_rsa

        if [[ "$(stat -c %U "$private/ssh/authorized_keys")" == "$USER" ]]; then
            symlink "$private/ssh/id_rsa.ppk" ~/.ssh/id_rsa.ppk
        else
            copy "$private/ssh/id_rsa.ppk" ~/.ssh/id_rsa.ppk
        fi

        if [[ "$(stat -c %U "$private/ssh/authorized_keys")" == "$USER" ]]; then
            symlink "$private/ssh/id_rsa.pub" ~/.ssh/id_rsa.pub
        else
            copy "$private/ssh/id_rsa.pub" ~/.ssh/id_rsa.pub
        fi
    fi
}


install_netdata() {
    if [[ "$(uname)" != "Darwin" ]]; then
        if ! test -f /usr/sbin/netdata ; then
            echo_ok "installing netdata"
            bash <(curl -Ss https://my-netdata.io/kickstart.sh) --dont-wait
            installed_netdata=1
        fi
    fi
}


install_dropbox() {
    mkdir -p ~/.local/bin
    symlink "$dotfiles/dropbox/dropbox.py" ~/.local/bin/dropbox

    if [[ "$(uname)" == "Darwin" ]]; then
        if ! grep "^dropbox$" < "$(brew_cask_list)" >/dev/null ; then
            brew_cask_install dropbox
            echo_error "wait for dropbox sync to complete then re-run"
            exit 0
        fi
    fi

    if [[ -d ~/Dropbos/Inbox ]]; then
        symlink Dropbox/Inbox ~/Inbox
    fi

    if [[ -d ~/Dropbox ]]; then
        symlink "$dotfiles/dropbox/dropbox-find-conflicts.sh" ~/.local/bin/dropbox-find-conflicts
    fi
}


install_fluid_apps() {
    if [[ "$(uname)" == "Darwin" ]]; then
        brew_cask_install fluid

        if [[ -d "$private/fluid.apps" ]]; then
            for app in $(find ~/Dropbox/Shared/fluid.apps -type d -depth 1 -name '*.app'); do
                symlink_dir "$app" "/Applications/$(basename "$app")"
            done
        fi
    fi
}


install_git() {
    _npm_install diff-so-fancy 0.11.4
    symlink .dotfiles/git/gitconfig ~/.gitconfig

    if [[ -d "$private/git" ]]; then
        symlink "$private/git/githubrc" ~/.githubrc
        symlink "$private/git/gogsrc" ~/.gogsrc
    fi
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
        if [[ -d "$private/ssmtp" ]]; then
            symlink "$private/ssmtp/ssmtp.conf" /etc/ssmtp/ssmtp.conf sudo
        fi
    fi
}


install_python() {
    # modern python
    if [[ "$(uname)" == "Darwin" ]]; then
        brew_install python
    else
        _apt_get_install python-pip
        _apt_get_install software-properties-common  # provides add-apt-repository

        if ! dpkg -s python3.6 &>/dev/null ; then
            sudo add-apt-repository -y ppa:jonathonf/python-3.6
            sudo apt-get update
            sudo apt-get install -y python3.6 python3.6-venv python3.6-dev python3-pip
        fi
    fi

    if [[ -f "$private/python/.pypirc" ]]; then
        symlink "$private/python/.pypirc" ~/.pypirc
    fi

    # install pip version
    local pip_version="9.0.1"
    if [[ "$(pip --version | awk '{print $2}')" != "$pip_version" ]]; then
        echo_ok "installing pip==$pip_version"
        pip install --upgrade "pip==$pip_version"
    fi
    # same again as root
    if [[ "$(sudo pip --version | awk '{print $2}')" != "$pip_version" ]]; then
        echo_ok "installing pip==$pip_version as root"
        sudo -H pip install --upgrade "pip==$pip_version"
    fi
}


install_unzip() {
    if [[ "$(uname)" != "Darwin" ]]; then
        _apt_get_install unzip
    fi
}


install_ruby() {
    if [[ "$(uname)" == "Darwin" ]]; then
        brew_install rbenv

        # initialize rbenv if required
        if $(which rbenv &>/dev/null); then
            eval "$(rbenv init -)"
        fi

        # install ruby and set as global version
        local ruby_version=2.4.1
        rbenv install --skip-existing "$ruby_version"
        rbenv global "$ruby_version"

        if ! gem list --local | grep bundler >/dev/null ; then
            echo_ok "gem install bundler"
            gem install bundler
        fi
    fi
}


install_curl() {
    if [[ "$(uname)" == "Darwin" ]]; then
        brew_install curl
    else
        _apt_get_install curl
    fi
}


install_node() {
    if [[ "$(uname)" == "Darwin" ]]; then
        brew_install node
    else
        _apt_get_install nodejs
        _apt_get_install npm
        symlink /usr/bin/nodejs /usr/bin/node sudo
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
    case "$hostname" in
      florence | diana | mary | plod)
        # server scripts
        symlink "$dotfiles/servers/mary" ~/.local/bin/mary
        symlink "$dotfiles/servers/diana" ~/.local/bin/diana
        ;;
    esac

    if [[ "$hostname" == "diana" ]]; then
        symlink "$dotfiles/servers/ryan_gosling_have_my_photos.sh" ~/.local/bin/ryan_gosling_have_my_photos
    fi

    if [[ "$hostname" == "florence" ]]; then
        symlink "$dotfiles/servers/ryan_gosling_have_my_music.sh" ~/.local/bin/ryan_gosling_have_my_music
    fi
}


install_macos() {
    # install Mac OS X specific stuff
    mkdir -p ~/.local/bin
    symlink "$dotfiles/macos/rm-dsstore" ~/.local/bin/rm-dsstore

    if [[ "$(uname)" == "Darwin" ]]; then
        # disable "Last Login ..." messages on terminal
        if [[ ! -f ~/.hushlogin ]]; then
            touch ~/.hushlogin
        fi
    fi
}


install_gpustat() {
    if which nvidia-smi >/dev/null ; then
        _pip_install gpustat 0.3.1
    fi
}


install_iotop() {
    _apt_get_install iotop
}


install_ncdu() {
    if [[ "$(uname)" == "Darwin" ]]; then
        brew_install ncdu
    else
        _apt_get_install ncdu
    fi
}


install_htop() {
    if [[ "$(uname)" == "Darwin" ]]; then
        brew_install htop
    else
        _apt_get_install htop
    fi
}


install_gh_archiver() {
    local gh_archiver_version=0.0.6

    if [[ "$(uname)" == "Darwin" ]]; then
        _pip_install gh-archiver $gh_archiver_version
    else
        _pip_install gh-archiver $gh_archiver_version "python3.6 -m pip"
    fi
}


install_emu() {
    # FIXME: pypi emu package not found
    # _pip_install emu 0.2.5 pip sudo
    true
}


install_json_util() {
    _npm_install jsonlint 1.6.2

    if [[ "$(uname)" == "Darwin" ]]; then
        brew_install jq
    else
        _apt_get_install jq
    fi
}


install_media() {
    mkdir -p ~/.local/bin
    # requires Python 3.6
    symlink "$dotfiles/media/mkepisodal.py" ~/.local/bin/mkepisodal
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


freeze_packages() {
    if [[ -d "$private/packages" ]]; then
        if [[ "$(uname)" == "Darwin" ]]; then
            cat "$(brew_list)" > "$private/packages/$hostname-brew.txt"
            cat "$(brew_cask_list)" > "$private/packages/$hostname-brew-cask.txt"
        fi

        cat "$(pip_freeze pip)" > "$private/packages/$hostname-pip.txt"
        cat "$(pip_freeze pip3)" > "$private/packages/$hostname-pip3.txt"
        cat "$(npm_list)" > "$private/packages/$hostname-npm.txt"
    fi
}


main() {
    parse_args $@
    echo_ok "dotfiles $(cat .install-lastrun.txt 2>/dev/null)..$(git rev-parse --short HEAD)"

    install_homebrew
    install_python
    install_unzip
    install_ruby
    install_curl
    install_dropbox
    install_fluid_apps
    install_ssh
    install_netdata
    install_node
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
    install_gpustat
    install_iotop
    install_ncdu
    install_htop
    install_gh_archiver
    install_emu
    install_json_util
    install_media

    # post-installation user prompts:
    if [[ "$installed_netdata" == "1" ]]; then
        echo_error "NOTE: manual steps required to complete netdata installation:"
        echo >&2
        echo "    $ crontab -e" >&2
        echo "    # append the following line to the file and save:" >&2
        echo "    @reboot ~/.dotfiles/crontab/start-netdata.sh" >&2
    fi

    freeze_packages

    git rev-parse --short HEAD > .install-lastrun.txt
}
main $@
