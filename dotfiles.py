#
# Copyright 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
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
import json

from util import *


class Apt(object):
    """ debian package manager, return True if installed """
    def install_package(self, package):
        """ install a package using apt-get """
        if not shell_ok("dpkg -s '{package}' &>/dev/null".format(**vars())):
            shell("sudo apt-get install -y '{package}'".format(**vars()))
            return True

    def update(self):
        """ update package information """
        shell("sudo apt-get update")


class Homebrew(Task):
    """ install homebrew package manager """
    # Temporary files for caching list of installed packages and casks
    PKG_LIST = os.path.abspath(".brew-pkgs.txt")
    CASK_LIST = os.path.abspath(".brew-casks.txt")
    OUTDATED_PKG_LIST = os.path.abspath(".brew-pkgs-outdated.txt")
    OUTDATED_CASK_LIST = os.path.abspath(".brew-casks-outdated.txt")
    __platforms__ = ['osx']
    __deps__ = []
    __genfiles__ = ['/usr/local/bin/brew']
    __tmpfiles__ = [PKG_LIST, CASK_LIST, OUTDATED_PKG_LIST, OUTDATED_CASK_LIST]

    def install(self):
        if not which('brew'):
            task_print("Installing Homebrew")
            shell('yes '' | /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"')
            shell('brew update')
            shell('brew doctor')

    def package_is_installed(self, package):
        """ return True if package is installed """
        if not os.path.isfile(self.PKG_LIST):
            shell("brew list > {self.PKG_LIST}".format(**vars()))

        return shell_ok("grep '^{package}$' <{self.PKG_LIST}".format(**vars()))

    def install_package(self, package):
        """ install a package using homebrew, return True if installed """
        if not self.package_is_installed(package):
            task_print("brew install " + package)
            shell("brew install {package}".format(**vars()))
            return True

    def package_is_outdated(self, package):
        """ returns True if package is outdated """
        if not self.package_is_installed(package):
            raise InvalidTaskError(
                "homebrew package '{package}' cannot be upgraded as it is not installed"
                .format(**vars()))

        if not os.path.isfile(self.OUTDATED_PKG_LIST):
            shell("brew outdated | awk '{{print $1}}' >{self.OUTDATED_PKG_LIST}"
                  .format(**vars()))

        package_stump = package.split('/')[-1]
        return shell_ok("grep '^{package_stump}$' <{self.OUTDATED_PKG_LIST}".format(**vars()))

    def upgrade_package(self, package):
        """ upgrade package, return True if upgraded """
        if self.package_is_outdated(package):
            task_print("brew upgrade {package}".format(**vars()))
            shell("brew upgrade {package}".format(**vars()))
            return True

    def cask_is_installed(self, cask):
        """ return True if cask is installed """
        if not os.path.isfile(self.CASK_LIST):
            shell("brew cask list > {self.CASK_LIST}".format(**vars()))

        cask_stump = cask.split('/')[-1]
        return shell_ok("grep '^{cask_stump}$' <{self.CASK_LIST}".format(**vars()))

    def install_cask(self, cask):
        """ install a homebrew cask, return True if installed """
        if not self.cask_is_installed(cask):
            task_print("brew cask install " + cask)
            shell("brew cask install {cask}".format(**vars()))
            return True

    def cask_is_outdated(self, cask):
        """ returns True if cask is outdated """
        if not self.cask_is_installed(cask):
            raise InvalidTaskError(
                "homebrew cask '{package}' cannot be upgraded as it is not installed"
               .format(**vars()))

        if not os.path.isfile(self.OUTDATED_CASK_LIST):
            shell("brew cask outdated | awk '{{print $1}}' >{self.OUTDATED_CASK_LIST}"
                  .format(**vars()))

        cask_stump = cask.split('/')[-1]
        return shell_ok("grep '^{cask_stump}$' <{self.OUTDATED_CASK_LIST}".format(**vars()))

    def upgrade_cask(self, cask):
        """ upgrade a homebrew cask. does nothing if cask not installed """
        if self.cask_is_outdated(cask):
            task_print("brew cask upgrade {cask}".format(**vars()))
            shell("brew cask upgrade {cask}".format(**vars()))
            return True

    def uninstall_cask(self, cask):
        """ remove a homebrew cask, return True if uninstalled """
        if self.cask_is_installed(cask):
            task_print("brew cask remove " + cask)
            shell("brew cask remove " + cask)
            return True


class Python(Task):
    """ python2 and pip """
    PYP_IRC = "~/.pypirc"
    PIP_LIST = ".pip-freeze.json"

    __platforms__ = ['linux', 'osx']
    __osx_deps__ = ['Homebrew']
    __genfiles__ = []
    __osx_genfiles__ = [
        '/usr/local/bin/pip2',
        '/usr/local/bin/pip3',
    ]
    __linux_genfiles__ = [
        '/usr/bin/pip3',
        '~/.local/bin/pip2',
    ]
    __tmpfiles__ = [PIP_LIST]
    __versions__ = {
        "pip": "9.0.1",
        "virtualenv": "15.1.0",
    }

    def install_osx(self):
        Homebrew().install_package("python")
        Homebrew().install_package("python3")
        self._install_common()

    def install_ubuntu(self):
        Apt().install_package("python-pip")
        Apt().install_package("python3-pip")
        Apt().install_package("software-properties-common")  # provides add-apt-repository
        if not shell_ok("dpkg -s python3.6 &>/dev/null"):
            task_print("adding python-3.6 repository")
            shell("sudo add-apt-repository -y ppa:jonathonf/python-3.6")
            shell("sudo apt-get update")
            task_print("apt-get install python3.6 python3.6-venv python3.6-dev python3-pip")
            shell("sudo apt-get install -y python3.6 python3.6-venv python3.6-dev python3-pip")

        self._install_common()

    def _pip_version_is(self, pip, version):
        return shell_ok(
            "test $({pip} --version | awk '{{print $2}}') = {version}"
            .format(**vars()))

    def _install_common(self):
        assert which("pip2")

        if os.path.isdir(PRIVATE):
            self.__genfiles__ += [self.PYP_IRC]
            symlink("{private}/python/.pypirc".format(private=PRIVATE),
                    "~/.pypirc")

        # install pip
        if not self._pip_version_is("pip2", self.__versions__["pip"]):
            task_print("pip2 install --upgrade 'pip=={pip_version}'".format(**vars()))
            shell("pip2 install --upgrade 'pip=={pip_version}'".format(**vars()))
        # same again as root
        if not self._pip_version_is("sudo pip2", self.__versions__["pip"]):
            shell("sudo -H pip2 install --upgrade 'pip=={pip_version}'".format(**vars()))

        # install pip
        if not self._pip_version_is("pip3", self.__versions__["pip"]):
            task_print("pip3 install --upgrade 'pip=={pip_version}'".format(**vars()))
            shell("pip3 install --upgrade 'pip=={pip_version}'".format(**vars()))
        # same again as root
        if not self._pip_version_is("sudo pip3", self.__versions__["pip"]):
            shell("sudo -H pip3 install --upgrade 'pip=={pip_version}'".format(**vars()))

        # install virtualenv
        self.pip_install("virtualenv", self.__versions__["virtualenv"])

    def upgrade_osx(self):
        Homebrew().upgrade_package("python")
        Homebrew().upgrade_package("python3")

    def pip_install(self, package, version, pip="pip2", sudo=False):
        """ install a package using pip, return True if installed """
        # Ubuntu requires sudo permission for pip install
        sudo = True if get_platform() == "ubuntu" else sudo
        use_sudo = "sudo -H " if sudo else ""

        # Create the list of pip packages
        if os.path.exists(self.PIP_LIST):
            with open(self.PIP_LIST) as infile:
                data = json.loads(infile.read())
        else:
            data = {}

        if pip not in data:
            freeze = shell("{use_sudo} {pip} freeze 2>/dev/null".format(**vars()))
            data[pip] = freeze.strip().split("\n")
            with open(self.PIP_LIST, "w") as outfile:
                json.dump(data, outfile)

        pkg_str = package + '==' + version
        if pkg_str not in data[pip]:
            task_print("pip install {package}=={version}".format(**vars()))
            shell("{use_sudo} {pip} install {package}=={version}".format(**vars()))
            return True


class Unzip(Task):
    """ unzip pacakge """
    __platforms__ = ['osx', 'ubuntu']
    __genfiles__ = ['/usr/bin/unzip']
    __osx_deps__ = ['Homebrew']

    def install_osx(self):
        Homebrew().install_package("unzip")

    def install_ubuntu(self):
        Apt().install_package("unzip")

    def upgrade_osx(self):
        Homebrew().upgrade_package("unzip")


class Ruby(Task):
    """ ruby environment """
    __platforms__ = ['osx']
    __osx_deps__ = ['Homebrew']
    __genfiles__ = ['~/.rbenv']
    __versions__ = {
        "ruby": "2.4.1"
    }

    def install_osx(self):
        Homebrew().install_package("rbenv")

        # initialize rbenv if required
        if shell_ok("which rbenv &>/dev/null"):
            shell('eval "$(rbenv init -)"')

        # install ruby and set as global version
        ruby_version = self.__versions__["ruby"]
        shell('rbenv install --skip-existing "{ruby_version}"'.format(**vars()))
        shell('rbenv global "{ruby_version}"'.format(**vars()))

        if not shell_ok("gem list --local | grep bundler"):
            task_print("gem install bundler")
            shell("sudo gem install bundler")

    def upgrade_osx(self):
        Homebrew().upgrade_package("rbenv")


class Curl(Task):
    """ curl command """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = ['Homebrew']
    __genfiles__ = ['/usr/bin/curl']

    def install_osx(self):
        Homebrew().install_package("curl")

    def install_ubuntu(self):
        Apt().install_package("curl")

    def upgrade_osx(self):
        Homebrew().upgrade_package("curl")


class Dropbox(Task):
    """ dropbox """
    UBUNTU_URL = "https://www.dropbox.com/download?plat=lnx.x86_64"

    __platforms__ = ['linux', 'osx']
    __osx_deps__ = ['Homebrew']
    __genfiles__ = ["~/.local/bin/dropbox"]
    __osx_genfiles__ = ["/Applications/Dropbox.app"]
    __linux_genfiles__ = ["~/.dropbox-dist/dropboxd"]

    def __init__(self):
        self.installed_on_ubuntu = False

    def _install_common(self):
        mkdir("~/.local/bin")
        symlink(usr_share("Dropbox/dropbox.py"), "~/.local/bin/dropbox")

        if (os.path.isdir(os.path.expanduser("~/Dropbox/Inbox")) and not
            os.path.isdir(os.path.expanduser("~/Dropbox/Inbox"))):
            self.__genfiles__.append("~/Inbox")
            symlink("Dropbox/Inbox", "~/Inbox")

        if os.path.isdir(os.path.expanduser("~/Dropbox")):
            self.__genfiles__.append("~/.local/bin/dropbox-find-conflicts")
            symlink(usr_share("Dropbox/dropbox-find-conflicts.sh"),
                    "~/.local/bin/dropbox-find-conflicts")

    def install_osx(self):
        Homebrew().install_cask("dropbox")
        self._install_common()

    def install_linux(self):
        if (not os.path.exists(os.path.expanduser("~/.dropbox-dist/dropboxd"))
            and not IS_TRAVIS_CI):  # skip on Travis CI:
            task_print("Installing Dropbox")
            shell('cd - && wget -O - "{self.UBUNTU_URL}" | tar xzf -'.format(**vars()))
            self.installed_on_ubuntu = True
        self._install_common()

    def upgrade_osx(self):
        Homebrew().upgrade_cask("dropbox")

    def teardown(self):
        if self.installed_on_ubuntu:
            logging.info("")
            logging.info("NOTE: manual step required to complete dropbox installation:")
            logging.info("    $ " + Colors.BOLD + Colors.RED +
                         "~/.dropbox-dist/dropboxd" + Colors.END)


class Fluid(Task):
    """ standalone web apps """
    __platforms__ = ['osx']
    __deps__ = ['Homebrew']
    __genfiles__ = ['/Applications/Fluid.app']

    def install_osx(self):
        Homebrew().install_cask("fluid")

        if os.path.isdir(PRIVATE + "/fluid.apps"):
            for app in os.listdir(PRIVATE + "/fluid.apps"):
                if app.endswith(".app"):
                    appname = os.path.basename(app)
                    if not os.path.exists("/Applications/" + os.path.basename(app)):
                        task_print("Installing {app}".format(**vars()))
                        shell("cp -r '{}/fluid.apps/{}' '/Applications/{}'"
                              .format(PRIVATE, app, app))

    def upgrade_osx(self):
        Homebrew().upgrade_cask("fluid")


class SSH(Task):
    """ ssh configuration """
    __platforms__ = ['linux', 'osx']
    __genfiles__ = []

    def install(self):
        if os.path.isdir(PRIVATE + "/ssh"):
            self.__genfiles__ += [
                "~/.ssh/authorized_keys",
                "~/.ssh/known_hosts",
                "~/.ssh/config",
                "~/.ssh/id_rsa.ppk",
                "~/.ssh/id_rsa.pub",
                "~/.ssh/id_rsa",
            ]

            mkdir("~/.ssh")
            shell('chmod 600 "' + PRIVATE + '"/ssh/*')

            for file in ['authorized_keys', 'known_hosts', 'config', 'id_rsa.ppk', 'id_rsa.pub']:
                src = os.path.join(PRIVATE, "ssh", file)
                dst = os.path.join("~/.ssh", file)

                if shell_ok("test $(stat -c %U '{src}') = $USER".format(**vars())):
                    symlink(src, dst)
                else:
                    copy_file(src, dst)
                    shell("chmod 600 {dst}".format(**vars()))

            copy_file(os.path.join(PRIVATE, "ssh", "id_rsa"), "~/.ssh/id_rsa")
            shell("chmod 600 ~/.ssh/id_rsa")


class Netdata(Task):
    """ realtime server monitoring """
    __platforms__ = ['ubuntu']
    __genfiles__ = ['/usr/sbin/netdata']

    def __init__(self):
        self.installed = False

    def install_linux(self):
        if not os.path.isfile("/usr/sbin/netdata"):
            task_print("Installing netdata")
            shell("bash <(curl -Ss https://my-netdata.io/kickstart.sh) --dont-wait")
            self.installed = True

    def teardown(self):
        if self.installed:
            logging.info("")
            logging.info("NOTE: manual steps required to complete netdata installation:")
            logging.info("    $ " + Colors.BOLD + Colors.RED + "crontab -e" + Colors.END)
            logging.info("    # append the following line to the end and save:")
            logging.info("    @reboot ~/.dotfiles/usr/share/crontab/start-netdata.sh")


class WacomDriver(Task):
    """ wacom tablet driver """
    __platforms__ = ['osx']
    __deps__ = ['Homebrew']
    __genfiles__ = ['/Applications/Wacom Tablet.localized']

    def __init__(self):
        self.installed = False

    def install(self):
        if Homebrew().install_cask('caskroom/drivers/wacom-intuos-tablet'):
            self.installed = True

    def upgrade(self):
        Homebrew().upgrade_cask("wacom-intuos-tablet")

    def teardown(self):
        if self.installed:
            logging.info("")
            logging.info("NOTE: manual steps required to complete Wacom driver setup:")
            logging.info("    " + Colors.BOLD + Colors.RED +
                         "Enable Wacom kernel extension in System Preferences > Security & Privacy" +
                         Colors.END)


class Node(Task):
    """ nodejs and npm """
    PKG_LIST = os.path.abspath(".npm-list.txt")

    NPM_BINARY = {
        "osx": "/usr/local/bin/npm",
        "ubuntu": "/usr/bin/npm"
    }

    __platforms__ = ['linux', 'osx']
    __osx_deps__ = ['Homebrew']
    __genfiles__ = [NPM_BINARY[PLATFORM]]
    __osx_genfiles__ = ['/usr/local/bin/node']
    __linux_genfiles__ = ['/usr/bin/node']
    __tmpfiles__ = [PKG_LIST]

    def install_osx(self):
        Homebrew().install_package("node")

    def install_ubuntu(self):
        Apt().install_package("nodejs")
        Apt().install_package("npm")
        symlink("/usr/bin/nodejs", "/usr/bin/node", sudo=True)

    def upgrade_osx(self):
        Homebrew().upgrade_package("node")

    def npm_install(self, package, version):
        """ install a package using npm, return True if installed """
        npm = self.NPM_BINARY[PLATFORM]

        # Create the list of npm packages
        if not os.path.isfile(self.PKG_LIST):
            shell("{npm} list -g > {self.PKG_LIST}".format(**vars()))

        if not shell_ok("grep '{package}@{version}' <{self.PKG_LIST}".format(**vars())):
            task_print("npm install -g {package}@{version}".format(**vars()))
            shell("sudo {npm} install -g {package}@{version}".format(**vars()))
            return True


class Zsh(Task):
    """ zsh shell and config files """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = ['Homebrew']
    __genfiles__ = [
        '~/.oh-my-zsh',
        '~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting',
        '~/.zsh',
        '~/.zsh/cec.zsh-theme',
        '~/.zshrc',
    ]
    __osx_genfiles__ = ['/usr/local/bin/zsh']
    __linux_genfiles__ = ['/usr/bin/zsh']
    __versions__ = {
        "oh-my-zsh": "c3b072eace1ce19a48e36c2ead5932ae2d2e06d9",
        "zsh-syntax-highlighting": "b07ada1255b74c25fbc96901f2b77dc4bd81de1a",
    }

    def install_osx(self):
        Homebrew().install_package("zsh")
        self.install()

    def install(self):
        # install config files
        symlink(usr_share("Zsh"), "~/.zsh")
        symlink(usr_share("Zsh/zshrc"), "~/.zshrc")
        if os.path.isdir(os.path.join(PRIVATE, "zsh")):
            self.__genfiles__ += ["~/.zsh/private"]
            symlink(os.path.join(PRIVATE, "zsh"), "~/.zsh/private")

        # oh-my-zsh
        clone_git_repo("git@github.com:robbyrussell/oh-my-zsh.git",
                       "~/.oh-my-zsh", self.__versions__["oh-my-zsh"])
        symlink("~/.zsh/cec.zsh-theme", "~/.oh-my-zsh/custom/cec.zsh-theme")

        # syntax highlighting module
        clone_git_repo("git@github.com:zsh-users/zsh-syntax-highlighting.git",
                       "~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting",
                       self.__versions__["zsh-syntax-highlighting"])

    def upgrade_osx(self):
        Homebrew().upgrade_package("zsh")


class Autoenv(Task):
    """ 'cd' wrapper """
    __platforms__ = ['linux', 'osx']
    __deps__ = ['Python']
    __genfiles__ = ['/usr/local/bin/activate.sh']

    def install(self):
        Python().pip_install("autoenv", "1.0.0")


class Lmk(Task):
    """ let-me-know """
    __platforms__ = ['linux', 'osx']
    __deps__ = ['Python']
    __genfiles__ = ['/usr/local/bin/lmk']
    __versions__ = {
        "lmk": "0.0.13"
    }

    def install(self):
        Python().pip_install("lmk", self.__versions__["lmk"])
        if os.path.isdir(os.path.join(PRIVATE, "lmk")):
            self.__genfiles__ += ["~/.lmkrc"]
            symlink(os.path.join(PRIVATE, "lmk", "lmkrc"), "~/.lmkrc")


class DSmith(Task):
    """ dsmith config """
    __platforms__ = ['ubuntu']
    __genfiles__ = []

    def install(self):
        if os.path.isdir(os.path.join(PRIVATE, "dsmith")):
            self.__genfiles__ += ['~/.dsmithrc']
            symlink(os.path.join(PRIVATE, "dsmith", "dsmithrc"), "~/.dsmithrc")


class Git(Task):
    """ git config """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = ['Homebrew']
    __genfiles__ = ['~/.gitconfig']

    def install_ubuntu(self):
        Apt().install_package('git')
        self.install()

    def install_osx(self):
        Homebrew().install_package('git')
        self.install()

    def install(self):
        if not IS_TRAVIS_CI:
            symlink(usr_share("git/gitconfig"), "~/.gitconfig")

        if os.path.isdir(os.path.join(PRIVATE, "git")):
            self.__genfiles__ += ['~/.githubrc', '~/.gogsrc']
            symlink(os.path.join(PRIVATE, "git", "githubrc"), "~/.githubrc")
            symlink(os.path.join(PRIVATE, "git", "gogsrc"), "~/.gogsrc")

    def upgrade_osx(self):
        Homebrew().upgrade_package("git")


class Wallpaper(Task):
    """ set desktop background """
    WALLPAPERS = {
        "diana": "~/Dropbox/Pictures/desktops/diana/Manhattan.jpg",
        "florence": "~/Dropbox/Pictures/desktops/florence/Uluru.jpg",
    }

    __platforms__ = ['osx']

    def install_osx(self):
        if HOSTNAME in self.WALLPAPERS:
            path = os.path.expanduser(self.WALLPAPERS[HOSTNAME])
            if os.path.exists(path):
                shell("osascript -e 'tell application \"Finder\" to set " +
                      "desktop picture to POSIX file \"{path}\"'"
                      .format(**vars()))


class GnuCoreutils(Task):
    """ replace BSD utils with GNU """
    __platforms__ = ['osx']
    __deps__ = ['Homebrew']
    __genfiles__ = [
        '/usr/local/opt/coreutils/libexec/gnubin/cp',
        '/usr/local/opt/gnu-sed/libexec/gnubin/sed',
        '/usr/local/opt/gnu-tar/libexec/gnubin/tar',
    ]

    def install(self):
        Homebrew().install_package('coreutils')
        Homebrew().install_package('gnu-indent')
        Homebrew().install_package('gnu-sed')
        Homebrew().install_package('gnu-tar')
        Homebrew().install_package('gnu-time')
        Homebrew().install_package('gnu-which')

    def upgrade(self):
        Homebrew().upgrade_package("coreutils")
        Homebrew().upgrade_package('gnu-indent')
        Homebrew().upgrade_package('gnu-sed')
        Homebrew().upgrade_package('gnu-tar')
        Homebrew().upgrade_package('gnu-time')
        Homebrew().upgrade_package('gnu-which')


class DiffSoFancy(Task):
    """ nice diff pager """
    VERSION = "0.11.4"

    __platforms__ = ['linux', 'osx']
    __deps__ = ['Git', 'Node']
    __genfiles__ = ['/usr/local/bin/diff-so-fancy']

    def install(self):
        Node().npm_install("diff-so-fancy", self.VERSION)


class GhArchiver(Task):
    """ github archiver """
    VERSION = "0.0.6"

    __platforms__ = ['linux', 'osx']
    __deps__ = ['Python']
    __genfiles__ = ['/usr/local/bin/gh-archiver']

    def install(self):
        Python().pip_install("gh-archiver", self.VERSION, pip="python3.6 -m pip")


class Tmux(Task):
    """ tmux config """
    __platforms__ = ['linux', 'osx']
    __genfiles__ = ['~/.tmux.conf']
    __osx_genfiles__ = ['/usr/local/bin/tmux']
    __linux_genfiles__ = ['/usr/bin/tmux']

    def install_osx(self):
        Homebrew().install_package("tmux")
        self._install_common()

    def install_ubuntu(self):
        Apt().install_package("tmux")
        self._install_common()

    def _install_common(self):
        symlink(usr_share("tmux/tmux.conf"), "~/.tmux.conf")

    def upgrade_osx(self):
        Homebrew().upgrade_package("tmux")


class Vim(Task):
    """ vim configuration """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = ['Homebrew']
    __genfiles__ = ['~/.vimrc', '~/.vim/bundle/Vundle.vim']
    __osx_genfiles__ = ['/usr/local/bin/vim']
    __linux_genfiles__ = ['/usr/bin/vim']
    __versions__ = {
        "vundle": "fcc204205e3305c4f86f07e09cd756c7d06f0f00",
    }

    def install_osx(self):
        Homebrew().install_package('vim')
        self.install()

    def install_ubuntu(self):
        Apt().install_package('vim')
        self.install()

    def install(self):
        symlink(usr_share("Vim/vimrc"), "~/.vimrc")

        # Vundle
        clone_git_repo("git@github.com:VundleVim/Vundle.vim.git",
                       "~/.vim/bundle/Vundle.vim",
                       self.__versions__["vundle"])
        shell("vim +PluginInstall +qall")

    def upgrade_osx(self):
        Homebrew().upgrade_package("vim")


class Linters(Task):
    __platforms__ = ['osx']
    __osx_deps__ = ['Node', 'Python']
    __osx_genfiles__ = [
        '/usr/local/bin/cpplint',
        '/usr/local/bin/csslint',
        '/usr/local/bin/pycodestyle',
        '~/.config/pycodestyle',
    ]
    __versions__ = {
        "cpplint": "1.3.0",
        "csslint": "1.0.5",
        "pycodestyle": "2.3.1",
    }

    def install_osx(self):
        Python().pip_install("cpplint", version=self.__versions__["cpplint"])
        Node().npm_install("csslint", version=self.__versions__["csslint"])
        Python().pip_install("pycodestyle", version=self.__versions__["2.3.1"],
                             pip="pip3.6")

        mkdir("~/.config")
        symlink(usr_share("linters/pycodestyle"), "~/.config/pycodestyle")


class SublimeText(Task):
    """ sublime text """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = ['Homebrew', 'Linters']
    __genfiles__ = ['/usr/local/bin/rsub']
    __osx_genfiles__ = ['/usr/local/bin/subl', '/Applications/Sublime Text.app']

    def install_osx(self):
        Homebrew().install_cask("sublime-text")

        # Put sublime text in PATH
        symlink("/Applications/Sublime Text.app/Contents/SharedSupport/bin/subl",
                "/usr/local/bin/subl", sudo=True)

        if os.path.isdir(os.path.join(PRIVATE, "subl")):
            self.__genfiles__ += [
                "~/.subl",
                "~/.subl/Packages/User",
                "~/.subl/Packages/INI"
            ]
            symlink("~/Library/Application Support/Sublime Text 3", "~/.subl")
            symlink(os.path.join(PRIVATE, "subl", "User"), "~/.subl/Packages/User")
            symlink(os.path.join(PRIVATE, "subl", "INI"), "~/.subl/Packages/INI")

        self.install()

    def install(self):
        symlink(usr_share("Sublime Text/rsub"), "/usr/local/bin/rsub", sudo=True)

    def upgrade_osx(self):
        Homebrew().upgrade_cask("sublime-text")


class Ssmtp(Task):
    """ mail server and config """
    __platforms__ = ['ubuntu']
    __genfiles__ = ["/usr/sbin/ssmtp"]

    def install_ubuntu(self):
        Apt().install_package("ssmtp")

        if os.path.isdir(os.path.join(PRIVATE, "ssmtp")):
            self.__genfiles__ += ["/etc/ssmtp/ssmtp.conf"]
            symlink(os.path.join(PRIVATE, "ssmtp", "ssmtp.conf"),
                    "/etc/ssmtp/ssmtp.conf", sudo=True)


class MySQL(Task):
    """ mysql configuration """
    __platforms__ = ['linux', 'osx']
    __genfiles__ = []

    def install(self):
        if os.path.isdir(os.path.join(PRIVATE, "mysql")):
            self.__genfiles__ += ["~/.my.cnf"]
            symlink(os.path.join(PRIVATE, "mysql", ".my.cnf"), "~/.my.cnf")


class LaTeX(Task):
    """ pdflatex and helper scripts """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = ['Homebrew']
    __genfiles__ = []
    __osx_genfiles__ = [
        '/Library/TeX/Distributions/.DefaultTeX/Contents/Programs/texbin/pdflatex',
        '/Applications/texstudio.app',
    ]

    def install_osx(self):
        Homebrew().install_cask("mactex")
        Homebrew().install_cask("texstudio")
        self.install()

    def install(self):
        if which("pdflatex"):
            self.__genfiles__ += ["~/.local/bin/autotex", "~/.local/bin/cleanbib"]
            mkdir("~/.local/bin")
            symlink(usr_share("LaTeX", "autotex"), "~/.local/bin/autotex")
            symlink(usr_share("LaTeX", "cleanbib"), "~/.local/bin/cleanbib")

    def upgrade(self):
        Homebrew().upgrade_cask("mactex")
        Homebrew().upgrade_cask("texstudio")


class AdobeCreativeCloud(Task):
    """ adobe creative cloud """
    __platforms__ = ['osx']
    __genfiles__ = [
        '/usr/local/Caskroom/adobe-creative-cloud/latest/Creative Cloud Installer.app',
        '/usr/local/Caskroom/google-nik-collection/1.2.11/Nik Collection.app',
    ]
    __deps__ = ['Homebrew']

    def __init__(self):
        self.installed = False

    def install(self):
        if not os.path.exists('/Applications/Adobe Lightroom CC/Adobe Lightroom CC.app'):
            Homebrew().install_cask('adobe-creative-cloud')
            self.installed = True
        if not os.path.exists('/Applications/Nik Collection'):
            Homebrew().install_cask('google-nik-collection')
            self.installed = True

    def upgrade(self):
        Homebrew().upgrade_cask("adobe-creative-cloud")
        Homebrew().upgrade_cask("google-nik-collection")

    def teardown(self):
        if self.installed:
            logging.info("")
            logging.info("NOTE: manual step required to complete creative cloud installation:")
            logging.info("    $ " + Colors.BOLD + Colors.RED +
                         "open '/usr/local/Caskroom/adobe-creative-cloud/latest/Creative Cloud Installer.app'" +
                         Colors.END)
            logging.info("    $ " + Colors.BOLD + Colors.RED +
                         "open '/usr/local/Caskroom/google-nik-collection/1.2.11/Nik Collection.app'" +
                         Colors.END)


class MacOSConfig(Task):
    """ macOS specific stuff """
    HUSHLOGIN = os.path.expanduser("~/.hushlogin")

    __platforms__ = ["osx"]
    __genfiles__ = ['~/.hushlogin']

    def install_osx(self):
        # disable "Last Login ..." messages on terminal
        if not os.path.exists(os.path.expanduser("~/.hushlogin")):
            task_print("Creating ~/.hushlogin")
            shell("touch " + os.path.expanduser("~/.hushlogin"))


class HomebrewCasks(Task):
    """ macOS homebrew binaries """
    CASKS = {
        'alfred': '/Applications/Alfred 3.app',
        'anki': '/Applications/Anki.app',
        'bartender': '/Applications/Bartender 3.app',
        'bettertouchtool': '/Applications/BetterTouchTool.app',
        'caffeine': '/Applications/Caffeine.app',
        'calibre': '/Applications/calibre.app',
        'dash': '/Applications/Dash.app',
        'disk-inventory-x': '/Applications/Disk Inventory X.app',
        'fantastical': '/Applications/Fantastical 2.app',
        'fluid': '/Applications/Fluid.app',
        'flux': '/Applications/Flux.app',
        'google-earth-pro': '/Applications/Google Earth Pro.app',
        'google-photos-backup-and-sync': '/Applications/Backup and Sync.app',
        'hipchat': '/Applications/HipChat.app',
        'imageoptim': '/Applications/ImageOptim.app',
        'istat-menus': '/Applications/iStat Menus.app',
        'iterm2': '/Applications/iTerm.app',
        'mendeley': '/Applications/Mendeley Desktop.app',
        'microsoft-office': '/Applications/Microsoft Word.app',
        'mysqlworkbench': '/Applications/MySQLWorkbench.app',
        'omnigraffle': '/Applications/OmniGraffle.app',
        'omnioutliner': '/Applications/OmniOutliner.app',
        'omnipresence': '/Applications/OmniPresence.app',
        'skype': '/Applications/Skype.app',
        'steam': '/Applications/Steam.app',
        'transmission': '/Applications/Transmission.app',
        'tunnelblick': '/Applications/Tunnelblick.app',
        'vlc': '/Applications/VLC.app',
    }

    __platforms__ = ['osx']
    __deps__ = ['Homebrew']
    __genfiles__ = list(CASKS.values())

    def install(self):
        for cask in self.CASKS.keys():
            Homebrew().install_cask(cask)

    def upgrade(self):
        for cask in self.CASKS.keys():
            Homebrew().upgrade_cask(cask)


class Plex(Task):
    """ plex media server and player """
    __platforms__ = ['osx']
    __deps__ = ['Homebrew']
    __genfiles__ = [
        '/Applications/Plex Media Player.app',
        '/Applications/Plex Media Server.app',
    ]

    def __init__(self):
        self.installed_server = False

    def install(self):
        Homebrew().install_cask('plex-media-player')
        self.installed_server = Homebrew().install_cask('plex-media-server')

    def upgrade(self):
        Homebrew().upgrade_cask('plex-media-player')
        Homebrew().upgrade_cask('plex-media-server')

    def uninstall(self):
        Homebrew().uninstall_cask('plrex-media-player')
        Homebrew().uninstall_cask('plrex-media-server')

    def teardown(self):
        if self.installed_server:
            logging.info("")
            logging.info("NOTE: manual step required to configure Plex Media Server:")
            logging.info("    $ " + Colors.BOLD + Colors.RED +
                         "open http://127.0.0.1:32400" + Colors.END)


class Trash(Task):
    VERSION = '1.4.0'

    __platforms__ = ['linux', 'osx']
    __deps__ = ['Node']
    __genfiles__ = ['/usr/local/bin/trash']

    def install(self):
        Node().npm_install('trash-cli', self.VERSION)

    def trash(self, *paths):
        for path in paths:
            path = os.path.expanduser(path)
            shell("/usr/local/bin/trash '{path}'".format(**vars()))


class Emacs(Task):
    """ emacs text editor and config """
    __platforms__ = ['linux', 'osx']
    __deps__ = ['Trash']
    __osx_deps__ = ['Homebrew']
    __versions__ = {
        "prelude": "f7d5d68d432319bb66a5f9410d2e4eadd584f498",
    }

    def install_osx(self):
        Homebrew().install_cask('emacs')
        self._install_common()

    def install_ubuntu(self):
        Apt().install_package('emacs')
        self._install_common()

    def _install_common(self):
        clone_git_repo("git@github.com:bbatsov/prelude.git",
                       "~/.emacs.d", self.__versions__["prelude"])
        # prelude requires there be no ~/.emacs file on first run
        Trash().trash('~/.emacs')

    def upgrade_osx(self):
        Homebrew().upgrade_cask('emacs')


class AppStore(Task):
    """ install mas app store command line """
    __platforms__ = ['osx']
    __deps__ = ['Homebrew']

    def install(self):
        Homebrew().install_package('mas')

        # Check that dotfiles was configured with Apple ID
        if not APPLE_ID:
            logging.critical("\nerror: Apple ID not set! Run ./configure --apple-id <email>")
            sys.exit(1)

        # Sign in to App Store using Apple ID. Note that if the user is
        # already signed in, this process will fail. We catch and ignore this
        # failure
        try:
            shell("mas signin --dialog " + APPLE_ID)
        except CalledProcessError:
            pass

    def upgrade(self):
        Homebrew().upgrade_package("mas")

    def install_app(self, package_id, package_dest):
        """ install package from App Store, return True if installed """
        if not os.path.exists(package_dest):
            package_stub = os.path.basename(package_dest)[:-4]
            task_print("mas install '{package_stub}'".format(**vars()))
            shell("mas install {package_id}".format(package_id=package_id))
            return True


class AppStoreApps(Task):
    """ install macOS apps from App Store """
    APPS = {
        443987910: '/Applications/1Password.app',
        961632517: '/Applications/Be Focused Pro.app',
        425264550: '/Applications/Blackmagic Disk Speed Test.app',
        420212497: '/Applications/Byword.app',
        563362017: '/Applications/CloudClip Manager.app',
        668208984: '/Applications/GIPHY CAPTURE.app',
        1026566364: '/Applications/GoodNotes.app',
        409183694: '/Applications/Keynote.app',
        784801555: '/Applications/Microsoft OneNote.app',
        823766827: '/Applications/OneDrive.app',
        409201541: '/Applications/Pages.app',
        425424353: '/Applications/The Unarchiver.app',
        1147396723: '/Applications/WhatsApp.app',
        410628904: '/Applications/Wunderlist.app',
    }

    HOST_APPS = {
        "diana": {
            883878097: '/Applications/Server.app'
        },
    }

    __platforms__ = ['osx']
    __deps__ = ['AppStore']
    __genfiles__ = list(APPS.values())

    def install(self):
        for app_id in self.APPS.keys():
            AppStore().install_app(app_id, self.APPS[app_id])

        for app_id in self.HOST_APPS.get(HOSTNAME, dict()):
            AppStore().install_app(app_id, self.HOST_APPS[HOSTNAME][app_id])


class GpuStat(Task):
    """ nice nvidia-smi wrapper """
    VERSION = "0.3.1"

    __platforms__ = ['linux', 'osx']
    __deps__ = ['Python']

    def install(self):
        if which("nvidia-smi"):
            Python().pip_install("gpustat", self.VERSION)


class IOTop(Task):
    """ I/O monitor """
    __platforms__ = ['ubuntu']
    __genfiles__ = ['/usr/sbin/iotop']

    def install_ubuntu(self):
        Apt().install_package("iotop")


class Ncdu(Task):
    """ cli disk space analyzer """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = ['Homebrew']
    __osx_genfiles__ = ['/usr/local/bin/ncdu']
    __linux_genfiles__ = ['/usr/bin/ncdu']

    def install_osx(self):
        Homebrew().install_package("ncdu")

    def install_ubuntu(self):
        Apt().install_package("ncdu")

    def upgrade_osx(self):
        Homebrew().upgrade_package("ncdu")


class HTop(Task):
    """ cli activity monitor """
    __platforms__ = ['linux']
    __genfiles__ = ['/usr/bin/htop']

    def install_ubuntu(self):
        Apt().install_package("htop")


class Java(Task):
    """ java8 runtime and compiler """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = ['Homebrew']
    __osx_genfiles__ = ['/usr/local/Caskroom/java8']
    __linux_genfiles__ = ['/usr/bin/java']

    def install_osx(self):
        Homebrew().install_cask('caskroom/versions/java8')

    def install_ubuntu(self):
        Apt().install_package("openjdk-8-jdk")

    def upgrade_osx(self):
        Homebrew().upgrade_cask('java8')


class OmniFocus(Task):
    """ task manager and utilities """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = ['Homebrew', 'Java']
    __genfiles__ = ["/usr/local/bin/omni"]
    __osx_genfiles__ = [
        "/Applications/OmniFocus.app",
        "/usr/local/opt/ofexport/bin/of2",
    ],
    __versions__ = {
        "ofexport": "1.0.20"
    }

    OFEXPORT_URL = "https://github.com/psidnell/ofexport2/archive/ofexport-v2-" + __versions__["ofexport"] + ".zip"

    def install_osx(self):
        Homebrew().install_cask('omnifocus')

        # Check that of2 is installed and is the correct version
        if (not os.path.exists("/usr/local/opt/ofexport/bin/of2") and
            shell("/usr/local/opt/ofexport/bin/of2 -h").split("\n")[2] != "Version: " + self.OFEXPORT2_VERSION):
            task_print("Downloading ofexport")
            shell("rm -rf /usr/local/opt/ofexport")
            url, ver = self.OFEXPORT_URL, self.__versions__["ofexport"]
            shell("wget {url} -O /tmp/ofexport.zip".format(**vars()))
            task_print("Installing ofexport")
            shell("unzip -o /tmp/ofexport.zip")
            shell("rm -f /tmp/ofexport.zip")
            shell("mv ofexport2-ofexport-v2-{ver} /usr/local/opt/ofexport".format(**vars()))

        # Run common-install commands:
        self.install()

    def install(self):
        symlink(usr_share("OmniFocus/omni"), "/usr/local/bin/omni", sudo=True)

    def upgrade_osx(self):
        Homebrew().upgrade_cask('omnifocus')


class Toggl(Task):
    """ time tracking app """
    __platforms__ = ['osx']
    __osx_genfiles__ = ['/Applications/TogglDesktop.app']
    __osx_deps__ = ['AppStore']

    def install_osx(self):
        AppStore().install_app('957734279', '/Applications/TogglDesktop.app')


class MeCsv(Task):
    """ me.csv health and time tracking """
    __platforms__ = ['osx']
    __genfiles__ = []
    __osx_deps__ = ['OmniFocus', 'Toggl']

    def install_osx(self):
        if os.path.isdir(os.path.join(PRIVATE, "me.csv")):
            self.__genfiles__ += [
                "~/.me.json",
                "~/me.csv"
            ]
            symlink(os.path.join(PRIVATE, "me.csv", "config.json"), "~/.me.json")
            symlink(os.path.join(PRIVATE, "me.csv", "data"), "~/me.csv")


class Bazel(Task):
    """ bazel build system """
    __platforms__ = ['linux', 'osx']
    __deps__ = ['Java']
    __osx_deps__ = ['Homebrew']
    __osx_genfiles__ = ['/usr/local/bin/bazel']
    __linux_genfiles__ = ['/usr/bin/bazel']

    def install_osx(self):
        Homebrew().install_package('bazel')

    def install_ubuntu(self):
        # See: https://docs.bazel.build/versions/master/install-ubuntu.html

        # Add Bazel distribution URY
        shell('echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list')
        shell('curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -')

        # Install and update Bazel
        Apt().update()
        Apt().install_package("bazel")

    def upgrade_osx(self):
        Homebrew().upgrade_package("bazel")


class CMake(Task):
    """ cmake build system """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = ['Homebrew']
    __osx_genfiles__ = ['/usr/local/bin/cmake']
    __linux_genfiles__ = ['/usr/bin/cmake']

    def install_osx(self):
        Homebrew().install_package('cmake')

    def install_ubuntu(self):
        Apt().install_package("cmake")

    def upgrade_osx(self):
        Homebrew().upgrade_package("cmake")


class Wget(Task):
    """ wget """
    __platforms__ = ['osx']
    __deps__ = ['Homebrew']
    __genfiles__ = ['/usr/local/bin/wget']

    def install(self):
        Homebrew().install_package('wget')

    def upgrade_osx(self):
        Homebrew().upgrade_package("wget")


class Protobuf(Task):
    """ protocol buffers """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = ['Homebrew']
    __osx_genfiles__ = ['/usr/local/bin/protoc']
    __linux_genfiles__ = ['/usr/bin/protoc']

    def install_osx(self):
        Homebrew().install_package('protobuf')

    def install_ubuntu(self):
        Apt().install_package("protobuf")

    def upgrade_osx(self):
        Homebrew().upgrade_package("protobuf")


class Sloccount(Task):
    """ source line count """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = ['Homebrew']
    __osx_genfiles__ = ['/usr/local/bin/sloccount']
    __linux_genfiles__ = ['/usr/bin/sloccount']

    def install_osx(self):
        Homebrew().install_package('sloccount')

    def install_ubuntu(self):
        Apt().install_package("sloccount")

    def upgrade_osx(self):
        Homebrew().upgrade_package("sloccount")


class Emu(Task):
    """ backup software """
    VERSION = "0.3.0"
    PIP = "pip3"

    __platforms__ = ['linux', 'osx']
    __deps__ = ['Python']
    __genfiles__ = ['/usr/local/bin/emu']

    def install(self):
        Python().pip_install("emu", self.VERSION, pip=self.PIP, sudo=True)


class JsonUtil(Task):
    """ json cli utils """
    __platforms__ = ['linux', 'osx']
    __deps__ = ['Node']
    __osx_deps__ = ['Homebrew']
    __genfiles__ = ['/usr/local/bin/jsonlint']
    __osx_genfiles__ = ['/usr/local/bin/jq']
    __linux_genfiles__ = ['/usr/bin/jq']
    __versions__ = {
        "jsonlint": "1.6.2",
    }

    def install_osx(self):
        Homebrew().install_package("jq")
        self._install_common()

    def install_ubuntu(self):
        Apt().install_package("jq")
        self._install_common()

    def _install_common(self):
        Node().npm_install("jsonlint", self.__versions__["jsonlint"])

    def upgrade_osx(self):
        Homebrew().upgrade_package("jq")


class Scripts(Task):
    """ scripts and utils """
    __platforms__ = ['linux', 'osx']
    __genfiles__ = [
        '~/.local/bin/mkepisodal',
        '~/.local/bin/rm-dsstore',
    ]

    def install(self):
        mkdir("~/.local/bin")
        symlink(usr_share("scripts/mkepisodal.py"), "~/.local/bin/mkepisodal")
        symlink(usr_share("scripts/rm-dsstore.sh"), "~/.local/bin/rm-dsstore")

        if HOSTNAME == "florence":
            self.__genfiles__ += [
                "~/.local/bin/orange_you_glad_you_backup",
                "~/.local/bin/ryan_gosling_have_my_movies",
                "~/.local/bin/ryan_gosling_have_my_music",
                "~/.local/bin/ryan_gosling_have_my_photos",
            ]
            symlink(usr_share("scripts/orange_you_glad_you_backup.sh"),
                    "~/.local/bin/orange_you_glad_you_backup")
            symlink(usr_share("scripts/ryan_gosling_have_my_photos.sh"),
                    "~/.local/bin/ryan_gosling_have_my_photos")
            symlink(usr_share("scripts/ryan_gosling_have_my_movies.sh"),
                    "~/.local/bin/ryan_gosling_have_my_movies")
            symlink(usr_share("scripts/ryan_gosling_have_my_music.sh"),
                    "~/.local/bin/ryan_gosling_have_my_music")

        if HOSTNAME == "diana":
            self.__genfiles__ += ["~/.local/bin/orange"]
            symlink(usr_share("scripts/orange.sh"), "~/.local/bin/orange")

    def uninstall(self):
        self.install()  # run to resolve dynamic genfiles
        task_print("Removing scripts")
        Trash().trash(*self.__genfiles__)


class Phd(Task):
    """ phd repo """
    __platforms__ = ['linux', 'osx']
    __genfiles__ = ['~/phd']

    def install(self):
        clone_git_repo("git@github.com:ChrisCummins/phd.git", "~/phd")


class DefaultApps(Task):
    """ set default applications for file extensions """
    __platforms__ = ['osx']
    __deps__ = [
        'AppStoreApps',
        'Homebrew',
        'HomebrewCasks',
        'LaTeX',
        'SublimeText',
    ]

    # run `duti -x <extension>` to show associated app
    FILE_ASSOCIATIONS = {
        "7z": "cx.c3.theunarchiver",
        "avi": "org.videolan.vlc",
        "c": "com.sublimetext.3",
        "cpp": "com.sublimetext.3",
        "cxx": "com.sublimetext.3",
        "gz": "cx.c3.theunarchiver",
        "markdown": "com.sublimetext.3",
        "md": "com.sublimetext.3",
        "mkv": "org.videolan.vlc",
        "mov": "org.videolan.vlc",
        "mp4": "org.videolan.vlc",
        "mpg": "org.videolan.vlc",
        "nfo": "com.sublimetext.3",
        "py": "com.sublimetext.3",
        "py": "com.sublimetext.3",
        "rar": "cx.c3.theunarchiver",
        "tex": "texstudio",
        "text": "com.sublimetext.3",
        "torrent": "org.m0k.transmission",
        "txt": "com.sublimetext.3",
        "xml": "com.sublimetext.3",
        "zip": "cx.c3.theunarchiver",
    }

    def install(self):
        Homebrew().install_package('duti')
        for extension in self.FILE_ASSOCIATIONS:
            app = self.FILE_ASSOCIATIONS[extension]
            shell('duti -s {app} .{extension} all'.format(**vars()))

    def upgrade(self):
        Homebrew().upgrade_package("duti")


class GoogleChrome(Task):
    __platforms__ = ['osx']
    __deps__ = ['Homebrew']

    def install_osx(self):
        Homebrew().install_cask('google-chrome')

    def upgrade_osx(self):
        Homebrew().upgrade_cask('google-chrome')
