from __future__ import print_function

import json

from util import *


class Homebrew(Task):
    """ install homebrew package manager """
    PKG_LIST = os.path.abspath(".brew-list.txt")
    CASK_LIST = os.path.abspath(".brew-cask-list.txt")

    __platforms__ = ['osx']
    __deps__ = []
    __genfiles__ = ['/usr/local/bin/brew']
    __tmpfiles__ = [PKG_LIST, CASK_LIST]

    def run(self):
        if not which('brew'):
            logging.info("installing homebrew")
            shell('/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"')
            shell('brew update')
            shell('brew doctore')

    def install(self, package):
        """ install a package using homebrew """
        # Create the list of homebrew packages
        if not os.path.isfile(self.PKG_LIST):
            shell("brew list > {self.PKG_LIST}".format(**vars()))

        if not shell_ok("grep '^{package}$' <{self.PKG_LIST} >/dev/null".format(**vars())):
            shell("brew install {package} >/dev/null".format(**vars()))

    def cask_install(self, package):
        """ install a homebrew cask """
        # Create the list of homebrew casks
        if not os.path.isfile(self.CASK_LIST):
            shell("brew cask list > {self.CASK_LIST}".format(**vars()))

        if not shell_ok("grep '^{package}$' <{self.CASK_LIST} >/dev/null".format(**vars())):
            shell("brew cask install {package} >/dev/null".format(**vars()))


class HomebrewCaskOutdated(Task):
    """ brew-cask-outdated script """
    VERSION = "2f08b5a76605fbfa9ab0caeb4868a29ef7e69bb1"
    BINPATH = "~/.local/bin/brew-cask-outdated"
    REMOTE_URL = "https://raw.githubusercontent.com/bgandon/brew-cask-outdated/" + VERSION + "/brew-cask-outdated.sh"

    __platforms__ = ['osx']
    __deps__ = [Homebrew]
    __genfiles__ = ['~/.local/bin/brew-cask-outdated']

    def run(self):
        if not which('brew-cask-outdated'):
            shell("mkdir -p ~/.local/bin")
            shell("curl {self.REMOTE_URL} 2>/dev/null > {self.BINPATH}".format(**vars()))
            shell('chmod +x {self.BINPATH}'.format(**vars()))


class Python(Task):
    """ python2 and pip """
    PIP_VERSION = "9.0.1"
    PYP_IRC = "~/.pypirc"
    PIP_LIST = ".pip-freeze.json"

    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]
    __genfiles__ = [PYP_IRC]
    __osx_genfiles__ = ['/usr/local/bin/pip2']
    __linux_genfiles__ = ['~/.local/bin/pip2']
    __tmpfiles__ = [PIP_LIST]

    def run_osx(self):
        Homebrew().install("python")
        self._run_common()

    def run_ubuntu(self):
        Apt().install("python-pip")
        Apt().install("software-properties-common")  # provides add-apt-repository
        if not shell_ok("dpkg -s python3.6 &>/dev/null"):
            shell("sudo add-apt-repository -y ppa:jonathonf/python-3.6")
            shell("sudo apt-get update")
            shell("sudo apt-get install -y python3.6 python3.6-venv python3.6-dev python3-pip")

        self._run_common()

    def _run_common(self):
        assert which("pip2")

        if os.path.exists("{private}/python/.pypirc".format(private=PRIVATE)):
            symlink("{private}/python/.pypirc".format(private=PRIVATE), "~/.pypirc")

        # install pip
        if not shell_ok("test $(pip2 --version | awk '{{print $2}}') = {self.PIP_VERSION}".format(**vars())):
            shell("pip2 install --upgrade 'pip=={self.PIP_VERSION}'".format(**vars()))
        # same again as root
        if not shell_ok("test $(sudo pip2 --version | awk '{{print $2}}') = {self.PIP_VERSION}".format(**vars())):
            shell("sudo -H pip2 install --upgrade 'pip=={self.PIP_VERSION}'".format(**vars()))

    def pip_install(self, package, version, pip="pip2", sudo=False):
        """ install a package using pip """
        use_sudo = "sudo -H " if sudo else ""

        # Create the list of pip packages
        if os.path.exists(self.PIP_LIST):
            with open(self.PIP_LIST) as infile:
                data = json.loads(infile.read())
        else:
            data = {}

        if pip not in data:
            freeze = shell_output("{use_sudo} {pip} freeze 2>/dev/null".format(**vars()))
            data[pip] = freeze.strip().split("\n")
            with open(self.PIP_LIST, "w") as outfile:
                json.dump(data, outfile)

        pkg_str = package + '==' + version
        if pkg_str not in data[pip]:
            shell("{use_sudo} {pip} install {package}=={version}".format(**vars()))


class Unzip(Task):
    """ unzip pacakge """
    __platforms__ = ['ubuntu']
    __genfiles__ = ['/usr/bin/unzip']

    def run_ubuntu(self):
        Apt().install("unzip")


class Ruby(Task):
    """ ruby environment """
    RUBY_VERSION = "2.4.1"

    __platforms__ = ['osx']
    __osx_deps__ = [Homebrew]
    __genfiles__ = ['~/.rbenv']

    def run_osx(self):
        Homebrew().install("rbenv")

        # initialize rbenv if required
        if shell_ok("which rbenv &>/dev/null"):
            shell('eval "$(rbenv init -)"')

        # install ruby and set as global version
        shell('rbenv install --skip-existing "{self.RUBY_VERSION}"'.format(**vars()))
        shell('rbenv global "{self.RUBY_VERSION}"'.format(**vars()))

        if not shell_ok("gem list --local | grep bundler >/dev/null"):
            shell("gem install bundler")


class Curl(Task):
    """ curl command """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]
    __genfiles__ = ['/usr/bin/curl']

    def run_osx(self):
        Homebrew().install("curl")

    def run_ubuntu(self):
        Apt().install("curl")


class Dropbox(Task):
    """ dropbox """
    UBUNTU_URL = "https://www.dropbox.com/download?plat=lnx.x86_64"

    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]
    __genfiles__ = ["~/.local/bin/dropbox"]
    __linux_genfiles__ = ["~/.dropbox-dist/dropboxd"]

    def __init__(self):
        self.installed = False

    def _run_common(self):
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

    def run_osx(self):
        Homebrew().cask_install("dropbox")
        self._run_common()

    def run_linux(self):
        if (not os.path.exists(os.path.expanduser("~/.dropbox-dist/dropboxd"))
            and not IS_TRAVIS_CI):  # skip on Travis CI:
            shell('cd - && wget -O - "{self.UBUNTU_URL}" | tar xzf -'.format(**vars()))
            self.installed = True
        self._run_common()

    def teardown(self):
        if self.installed:
            print("NOTE: manual step required to complete dropbox installation:")
            print()
            print("    $ ~/.dropbox-dist/dropboxd")


class Fluid(Task):
    """ standalone web apps """
    __platforms__ = ['osx']
    __deps__ = [Homebrew]
    __genfiles__ = ['/Applications/Fluid.app']

    def run_osx(self):
        Homebrew().cask_install("fluid")

        if os.path.isdir(PRIVATE + "/fluid.apps"):
            for app in os.listdir(PRIVATE + "/fluid.apps"):
                if app.endswith(".app"):
                    if not os.path.exists("/Applications/" + os.path.basename(app)):
                        shell("cp -r '{}' '{}'".format(
                                PRIVATE + "/fluid.apps/" + app,
                                "/Applications/" + os.path.basename(app)))


class SSH(Task):
    """ ssh configuration """
    __platforms__ = ['linux', 'osx']
    __genfiles__ = []

    def run(self):
        if os.path.isdir(PRIVATE + "/ssh"):
            self.__genfiles__ += [
                "~/.ssh/authorized_keys",
                "~/.ssh/known_hosts",
                "~/.ssh/config",
                "~/.ssh/id_rsa.ppk",
                "~/.ssh/id_rsa.pub",
                "~/.ssh/id_rsa",
            ]

            shell('chmod 600 "' + PRIVATE + '"/ssh/*')
            mkdir("~/.ssh")

            for file in ['authorized_keys', 'known_hosts', 'config', 'id_rsa.ppk', 'id_rsa.pub']:
                src = os.path.join(PRIVATE, "ssh", file)
                dst = os.path.join("~/.ssh", file)

                if shell_ok("test $(stat -c %U '{src}') = $USER".format(**vars())):
                    symlink(src, dst)
                else:
                    copy(src, dst)

            copy(os.path.join(PRIVATE, "ssh", "id_rsa"), "~/.ssh/id_rsa")


class Netdata(Task):
    """ realtime server monitoring """
    __platforms__ = ['ubuntu']
    __genfiles__ = ['/usr/sbin/netdata']

    def __init__(self):
        self.installed = False

    def run_linux(self):
        if not os.path.isfile("/usr/sbin/netdata"):
            shell("bash <(curl -Ss https://my-netdata.io/kickstart.sh) --dont-wait")
            self.installed = True

    def teardown(self):
        if self.installed:
            print("NOTE: manual steps required to complete netdata installation:")
            print()
            print("    $ crontab -e")
            print("    # append the following line to the end and save:")
            print("    @reboot ~/.dotfiles/crontab/start-netdata.sh")


class Node(Task):
    """ nodejs and npm """
    PKG_LIST = os.path.abspath(".npm-list.txt")

    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]
    __osx_genfiles__ = ['/usr/local/bin/node', '/usr/local/bin/npm']
    __linux_genfiles__ = ['/usr/bin/node', '/usr/bin/npm']
    __tmpfiles__ = [PKG_LIST]

    def run_osx(self):
        Homebrew().install("node")

    def run_ubuntu(self):
        Apt().install("nodejs")
        Apt().install("npm")
        symlink("/usr/bin/nodejs", "/usr/bin/node", sudo=True)

    def npm_install(self, package, version):
        """ install a package using npm """
        # Create the list of npm packages
        if not os.path.isfile(self.PKG_LIST):
            shell("npm list -g > {self.PKG_LIST}".format(**vars()))

        if not shell_ok("grep '{package}@{version}' <{self.PKG_LIST} >/dev/null".format(**vars())):
            shell("sudo npm install -g {package}@{version}".format(**vars()))


class Zsh(Task):
    """ zsh and config files """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]
    __genfiles__ = ['~/.zshrc', '~/.zsh']
    __osx_genfiles__ = ['/usr/local/bin/zsh']
    __linux_genfiles__ = ['/usr/bin/zsh']

    def run_osx(self):
        Homebrew().install("zsh")
        self.run()

    def run(self):
        # install config files
        symlink(usr_share("Zsh"), "~/.zsh")
        symlink(usr_share("Zsh/zshrc"), "~/.zshrc")
        if os.path.isdir(os.path.join(PRIVATE, "zsh")):
            self.__genfiles__ += ["~/.zsh/private"]
            symlink(os.path.join(PRIVATE, "zsh"), "~/.zsh/private")


class Autoenv(Task):
    """ 'cd' wrapper """
    __platforms__ = ['linux', 'osx']
    __deps__ = [Python]
    __genfiles__ = ['/usr/local/bin/activate.sh']

    def run(self):
        Python().pip_install("autoenv", "1.0.0")


class OhMyZsh(Task):
    """ oh-my-zsh shell framework """
    __platforms__ = ['linux', 'osx']
    __deps__ = [Zsh]
    __genfiles__ = [
        '~/.oh-my-zsh',
        '~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting',
        '~/.zsh/cec.zsh-theme'
    ]

    def run(self):
        clone_git_repo("git@github.com:robbyrussell/oh-my-zsh.git",
                       "~/.oh-my-zsh",
                       "66bae5a5deb7a053adfb05b38a93fe47295841eb")

        # syntax highlighting
        clone_git_repo("git@github.com:zsh-users/zsh-syntax-highlighting.git",
                       "~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting",
                       "ad522a091429ba180c930f84b2a023b40de4dbcc")

        # oh-my-zsh config
        symlink("~/.zsh/cec.zsh-theme", "~/.oh-my-zsh/custom/cec.zsh-theme")


class Lmk(Task):
    """ let-me-know """
    LMK_VERSION = "0.0.13"

    __platforms__ = ['linux', 'osx']
    __deps__ = [Python]
    __genfiles__ = ['/usr/local/bin/lmk']

    def run(self):
        Python().pip_install("lmk", self.LMK_VERSION)
        if os.path.isdir(os.path.join(PRIVATE, "lmk")):
            self.__genfiles__ += ["~/.lmkrc"]
            symlink(os.path.join(PRIVATE, "lmk", "lmkrc"), "~/.lmkrc")


class DSmith(Task):
    """ dsmith config """
    __platforms__ = ['ubuntu']
    __genfiles__ = []

    def run(self):
        if os.path.isdir(os.path.join(PRIVATE, "dsmith")):
            self.__genfiles__ += ['~/.dsmithrc']
            symlink(os.path.join(PRIVATE, "dsmith", "dsmithrc"), "~/.dsmithrc")


class Git(Task):
    """ git config """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]
    __genfiles__ = ['~/.gitconfig']

    def run_ubuntu(self):
        Apt().install('git')
        self.run()

    def run_osx(self):
        Homebrew().install('git')
        self.run()

    def run(self):
        if not IS_TRAVIS_CI:
            symlink(usr_share("git/gitconfig"), "~/.gitconfig")

        if os.path.isdir(os.path.join(PRIVATE, "git")):
            self.__genfiles__ += ['~/.githubrc', '~/.gogsrc']
            symlink(os.path.join(PRIVATE, "git", "githubrc"), "~/.githubrc")
            symlink(os.path.join(PRIVATE, "git", "gogsrc"), "~/.gogsrc")


class DiffSoFancy(Task):
    """ nice diff pager """
    VERSION = "0.11.4"

    __platforms__ = ['linux', 'osx']
    __deps__ = [Git, Node]
    __genfiles__ = ['/usr/local/bin/diff-so-fancy']

    def run(self):
        Node().npm_install("diff-so-fancy", self.VERSION)


class GhArchiver(Task):
    """ github archiver """
    VERSION = "0.0.6"

    __platforms__ = ['linux', 'osx']
    __deps__ = [Python]
    __genfiles__ = ['/usr/local/bin/gh-archiver']

    def run(self):
        Python().pip_install("gh-archiver", self.VERSION, pip="python3.6 -m pip")


class Tmux(Task):
    """ tmux config """
    __platforms__ = ['linux', 'osx']
    __genfiles__ = ['~/.tmux.conf']

    def run(self):
        symlink(usr_share("tmux/tmux.conf"), "~/.tmux.conf")


class Vim(Task):
    """ vim configuration """
    VUNDLE_VERSION = "6497e37694cd2134ccc3e2526818447ee8f20f92"

    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]
    __genfiles__ = ['~/.vimrc', '~/.vim/bundle/Vundle.vim']
    __osx_genfiles__ = ['/usr/local/bin/vim']
    __linux_genfiles__ = ['/usr/bin/vim']

    def run_osx(self):
        Homebrew().install('vim')
        self.run()

    def run_ubuntu(self):
        Apt().install('vim')
        self.run()

    def run(self):
        symlink(usr_share("Vim/vimrc"), "~/.vimrc")

        # Vundle
        clone_git_repo("git@github.com:VundleVim/Vundle.vim.git",
                       "~/.vim/bundle/Vundle.vim",
                       self.VUNDLE_VERSION)
        shell("vim +PluginInstall +qall")


class Sublime(Task):
    """ sublime text """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]
    __genfiles__ = ['/usr/local/bin/rsub']
    __osx_genfiles__ = ['/usr/local/bin/subl', '/Applications/Sublime Text.app']

    def run_osx(self):
        self.run()
        Homebrew().cask_install("sublime-text")

        # Put sublime text in PATH
        symlink("/Applications/Sublime Text.app/Contents/SharedSupport/bin/subl",
                "/usr/local/bin/subl", sudo=True)

        if os.path.isdir(os.path.join(PRIVATE, "subl")):
            self.__genfiles__ += ["~/.subl", "~/.subl/Packages/User", "~/.subl/Packages/INI"]
            symlink("Library/Application Support/Sublime Text 3", "~/.subl")
            symlink(os.path.join(PRIVATE, "subl", "User"), "~/.subl/Packages/User")
            symlink(os.path.join(PRIVATE, "subl", "INI"), "~/.subl/Packages/INI")

    def run(self):
        shell('sudo ln -sf "{}" /usr/local/bin/rsub'.format(usr_share("Sublime Text/rsub")))


class Ssmtp(Task):
    """ mail server and config """
    __platforms__ = ['ubuntu']
    __genfiles__ = ["/usr/sbin/ssmtp"]

    def run_ubuntu(self):
        Apt().install("ssmtp")

        if os.path.isdir(os.path.join(PRIVATE, "ssmtp")):
            self.__genfiles__ += ["/etc/ssmtp/ssmtp.conf"]
            symlink(os.path.join(PRIVATE, "ssmtp", "ssmtp.conf"),
                    "/etc/ssmtp/ssmtp.conf", sudo=True)


class MySQL(Task):
    """ mysql configuration """
    __platforms__ = ['linux', 'osx']
    __genfiles__ = []

    def run(self):
        if os.path.isdir(os.path.join(PRIVATE, "mysql")):
            self.__genfiles__ += ["~/.my.cnf"]
            symlink(os.path.join(PRIVATE, "mysql", ".my.cnf"), "~/.my.cnf")


class OmniFocus(Task):
    __platforms__ = ['linux', 'osx']
    __genfiles__ = ["/usr/local/bin/omni"]

    def run(self):
        shell('sudo ln -sf "{}" /usr/local/bin'.format(usr_share("OmniFocus/omni")))


class LaTeX(Task):
    """ pdflatex and helper scripts """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]
    __genfiles__ = []

    def run_osx(self):
        self.__osx_genfiles__ = [
            '/Library/TeX/Distributions/.DefaultTeX/Contents/Programs/texbin/pdflatex'
        ]
        Homebrew().cask_install("mactex")
        self.run()

    def run(self):
        if which("pdflatex"):
            self.__genfiles__ += ["~/.local/bin/autotex", "~/.local/bin/cleanbib"]
            mkdir("~/.local/bin")
            symlink(usr_share("LaTeX", "autotex"), "~/.local/bin/autotex")
            symlink(usr_share("LaTeX", "cleanbib"), "~/.local/bin/cleanbib")


class MacOSConfig(Task):
    """ macOS specific stuff """
    HUSHLOGIN = os.path.expanduser("~/.hushlogin")

    __platforms__ = ["osx"]
    __genfiles__ = ['~/.hushlogin']

    def run_osx(self):
        # disable "Last Login ..." messages on terminal
        if not os.path.exists(os.path.expanduser("~/.hushlogin")):
            shell("touch " + os.path.expanduser("~/.hushlogin"))


class MacOSApps(Task):
    """ macOS applications """
    CASKS = {
        'alfred': '/Applications/Alfred 3.app',
        'anki': '/Applications/Anki.app',
        'bartender': '/Applications/Bartender 3.app',
        'caffeine': '/Applications/Caffeine.app',
        'cmake': '/Applications/CMake.app',
        'disk-inventory-x': '/Applications/Disk Inventory X.app',
        'fantastical': '/Applications/Fantastical 2.app',
        'flux': '/Applications/Flux.app',
        'google-earth-pro': '/Applications/Google Earth Pro.app',
        'google-nik-collection': '/Applications/Nik Collection',
        'google-photos-backup-and-sync': '/Applications/Backup and Sync.app',
        'istat-menus': '/Applications/iStat Menus.app',
        'iterm2': '/Applications/iTerm.app',
        'mendeley': '/Applications/Mendeley Desktop.app',
        'omnifocus': '/Applications/OmniFocus.app',
        'omnigraffle': '/Applications/OmniGraffle.app',
        'omnioutliner': '/Applications/OmniOutliner.app',
        'omnipresence': '/Applications/OmniPresence.app',
        'plex-media-player': '/Applications/Plex Media Player.app',
        'steam': '/Applications/Steam.app',
        'texstudio': '/Applications/texstudio.app',
        'transmission': '/Applications/Transmission.app',
        'tunnelblick': '/Applications/Tunnelblick.app',
        'vlc': '/Applications/VLC.app',
    }

    __platforms__ = ['osx']
    __deps__ = [Homebrew]
    __genfiles__ = list(CASKS.values())

    def run(self):
        for cask in self.CASKS.keys():
            Homebrew().cask_install(cask)


class GpuStat(Task):
    """ nice nvidia-smi wrapper """
    VERSION = "0.3.1"

    __platforms__ = ['linux', 'osx']
    __deps__ = [Python]

    def run(self):
        if which("nvidia-smi"):
            Python().pip_install("gpustat", self.VERSION)


class IOTop(Task):
    """ I/O monitor """
    __platforms__ = ['ubuntu']
    __genfiles__ = ['/usr/sbin/iotop']

    def run_ubuntu(self):
        Apt().install("iotop")


class Ncdu(Task):
    """ cli disk space analyzer """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]
    __osx_genfiles__ = ['/usr/local/bin/ncdu']
    __linux_genfiles__ = ['/usr/bin/ncdu']

    def run_osx(self):
        Homebrew().install("ncdu")

    def run_ubuntu(self):
        Apt().install("ncdu")


class HTop(Task):
    """ cli activity monitor """
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]
    __osx_genfiles__ = ['/usr/local/bin/htop']
    __linux_genfiles__ = ['/usr/bin/htop']

    def run_osx(self):
        Homebrew().install("htop")

    def run_ubuntu(self):
        Apt().install("htop")


class Emu(Task):
    """ backup software """
    VERSION = "0.3.0"
    PIP = "pip3"

    __platforms__ = ['linux', 'osx']
    __deps__ = [Python]
    __genfiles__ = ['/usr/local/bin/emu']

    def run(self):
        Python().pip_install("emu", self.VERSION, pip=self.PIP, sudo=True)


class JsonUtil(Task):
    """ json cli utils """
    JSONLINT_VERSION = "1.6.2"

    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]
    __genfiles__ = ['/usr/local/bin/jsonlint']
    __osx_genfiles__ = ['/usr/local/bin/jq']
    __linux_genfiles__ = ['/usr/bin/jq']

    def run_osx(self):
        Homebrew().install("jq")
        self._run_common()

    def run_ubuntu(self):
        Apt().install("jq")
        self._run_common()

    def _run_common(self):
        Node().npm_install("jsonlint", self.JSONLINT_VERSION)


class Scripts(Task):
    """ scripts and utils """
    __platforms__ = ['linux', 'osx']
    __genfiles__ = ['~/.local/bin/mkepisodal']

    def run(self):
        symlink(usr_share("media/mkepisodal.py"), "~/.local/bin/mkepisodal")

        if HOSTNAME in ["florence", "diana", "ryangosling", "mary", "plod"]:
            self.__genfiles__ += ["~/.local/bin/mary", "~/.local/bin/diana"]
            symlink(usr_share("servers/mary"), "~/.local/bin/mary")
            symlink(usr_share("servers/diana"), "~/.local/bin/diana")

        if HOSTNAME in ["florence", "diana"]:
            self.__genfiles__ += [
                "~/.local/bin/ryan_gosling_have_my_photos",
                "~/.local/bin/ryan_gosling_have_my_music",
                "~/.local/bin/orange",
            ]
            symlink(usr_share("servers/ryan_gosling_have_my_photos.sh"),
                    "~/.local/bin/ryan_gosling_have_my_photos")
            symlink(usr_share("servers/ryan_gosling_have_my_music.sh"),
                    "~/.local/bin/ryan_gosling_have_my_music")
            symlink(usr_share("servers/orange.sh"),
                    "~/.local/bin/orange")
