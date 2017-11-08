from __future__ import print_function

from dotfiles import *


class Homebrew(Task):
    PKG_LIST = os.path.abspath(".brew-list.txt")
    CASK_LIST = os.path.abspath(".brew-cask-list.txt")

    __platforms__ = ['osx']
    __deps__ = []
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


class Apt(object):
    def install(self, package):
        """ install a package using apt-get """
        if not shell_ok("dpkg -s '{package}' &>/dev/null".format(**vars())):
            shell("sudo apt-get install -y '{package}'".format(**vars()))


class HomebrewCaskOutdated(Task):
    VERSION = "2f08b5a76605fbfa9ab0caeb4868a29ef7e69bb1"
    BINPATH = "~/.local/bin/brew-cask-outdated"
    REMOTE_URL = "https://raw.githubusercontent.com/bgandon/brew-cask-outdated/" + VERSION + "/brew-cask-outdated.sh"

    __platforms__ = ['osx']
    __deps__ = [Homebrew]
    __installfiles__ = []

    def run(self):
        if not which('brew-cask-outdated'):
            shell("curl {self.REMOTE_URL} 2>/dev/null > {self.BINPATH}".format(**vars()))
            shell('chmod +x {self.BINPATH}'.format(**vars()))


class Python(Task):
    PIP_VERSION = "9.0.1"
    PYP_IRC = "~/.pypirc"
    PIP_LIST = ".pip-freeze.json"

    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]
    __genfiles__ = [PYP_IRC]
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
    __platforms__ = ['ubuntu']

    def run_ubuntu(self):
        Apt().install("unzip")


class Ruby(Task):
    RUBY_VERSION = "2.4.1"

    __platforms__ = ['osx']
    __osx_deps__ = [Homebrew]

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
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]

    def run_osx(self):
        Homebrew().install("curl")

    def run_ubuntu(self):
        Apt().install("curl")


class Dropbox(Task):
    UBUNTU_URL = "https://www.dropbox.com/download?plat=lnx.x86_64"

    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]

    __genfiles__ = ["~/.local/bin/dropbox-find-conflicts"]

    def __init__(self):
        self.installed = False

    def _run_common(self):
        mkdir("~/.local/bin")
        symlink("{df}/dropbox/dropbox.py".format(df=DOTFILES), "~/.local/bin/dropbox")

        if not os.path.isdir(os.path.expanduser("~/Dropbox/Inbox")):
            symlink("Dropbox/Inbox", "~/Inbox")

        if os.path.isdir(os.path.expanduser("~/Dropbox")):
            symlink("{df}/dropbox/dropbox-find-conflicts.sh".format(df=DOTFILES),
                    "~/.local/bin/dropbox-find-conflicts")

    def run_osx(self):
        Homebrew().cask_install("dropbox")
        self._run_common()

    def run_linux(self):
        if not os.path.exists(os.path.expanduser("~/.dropbox-dist/dropboxd")):
            shell('cd - && wget -O - "{self.UBUNTU_URL}" | tar xzf -')
            self.installed = True
        self._run_common()

    def teardown(self):
        if self.installed:
            print("NOTE: manual step required to complete dropbox installation:")
            print()
            print("    $ ~/.dropbox-dist/dropboxd")


class Fluid(Task):
    __platforms__ = ['osx']
    __deps__ = [Dropbox]
    __osx_deps__ = [Homebrew]

    def run_osx(self):
        Homebrew().cask_install("fluid")

        if os.path.isdir(PRIVATE + "/fluid.apps"):
            for app in os.listdir(PRIVATE + "/fluid.apps"):
                if app.endswith(".app"):
                    symlink(PRIVATE + "/fluid.apps/" + app, "/Applications/" + os.path.basename(app))


class SSH(Task):
    __platforms__ = ['linux', 'osx']
    __deps__ = [Dropbox]
    __genfiles__ = [
        "~/.ssh/authorized_keys",
        "~/.ssh/known_hosts",
        "~/.ssh/config",
        "~/.ssh/id_rsa.ppk",
        "~/.ssh/id_rsa.pub",
        "~/.ssh/id_rsa",
    ]

    def run(self):
        if os.path.isdir(PRIVATE + "/ssh"):
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
    __platforms__ = ['ubuntu']

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
    PKG_LIST = os.path.abspath(".npm-list.txt")

    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]
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
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]

    def run_osx(self):
        Homebrew().install("zsh")
        self.run()

    def run(self):
        # install config files
        symlink(os.path.join(DOTFILES, "zsh"), "~/.zsh")
        symlink(".zsh/zshrc", "~/.zshrc")
        if os.path.isdir(os.path.join(PRIVATE, "zsh")):
            symlink(os.path.join(PRIVATE, "zsh"), "~/.zsh/private")


class Autoenv(Task):
    __platforms__ = ['linux', 'osx']
    __deps__ = [Python]

    def run(self):
        Python().pip_install("autoenv", "1.0.0")


class OhMyZsh(Task):
    __platforms__ = ['linux', 'osx']
    __deps__ = [Zsh]

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
    LMK_VERSION = "0.0.13"

    __platforms__ = ['linux', 'osx']
    __deps__ = [Python]

    def run(self):
        Python().pip_install("lmk", self.LMK_VERSION)
        if os.path.isdir(os.path.join(PRIVATE, "lmk")):
            symlink(os.path.join(PRIVATE, "lmk", "lmkrc"), "~/.lmkrc")


class DSmith(Task):
    __platforms__ = ['ubuntu']

    def run(self):
        if os.path.isdir(os.path.join(PRIVATE, "dsmith")):
            symlink(os.path.join(PRIVATE, "dsmith", "dsmithrc"), "~/.dsmithrc")


class Git(Task):
    __platforms__ = ['linux', 'osx']

    def run(self):
        symlink(".dotfiles/git/gitconfig", "~/.gitconfig")

        if os.path.isdir(os.path.join(PRIVATE, "git")):
            symlink(os.path.join(PRIVATE, "git", "githubrc"), "~/.githubrc")
            symlink(os.path.join(PRIVATE, "git", "gogsrc"), "~/.gogsrc")


class DiffSoFancy(Task):
    VERSION = "0.11.4"

    __platforms__ = ['linux', 'osx']
    __deps__ = [Git, Node]

    def run(self):
        Node().npm_install("diff-so-fancy", self.VERSION)


class GhArchiver(Task):
    VERSION = "0.0.6"

    __platforms__ = ['linux', 'osx']
    __deps__ = [Python]

    def run(self):
        Python().pip_install("gh-archiver", self.VERSION, pip="python3.6 -m pip")


class Tmux(Task):
    __platforms__ = ['linux', 'osx']

    def run(self):
        symlink(".dotfiles/tmux/tmux.conf", "~/.tmux.conf")


class Vim(Task):
    VUNDLE_VERSION = "6497e37694cd2134ccc3e2526818447ee8f20f92"

    __platforms__ = ['linux', 'osx']

    def run(self):
        symlink(os.path.join(DOTFILES, "vim", "vimrc"), "~/.vimrc")

        # Vundle
        clone_git_repo("git@github.com:VundleVim/Vundle.vim.git",
                       "~/.vim/bundle/Vundle.vim",
                       self.VUNDLE_VERSION)
        shell("vim +PluginInstall +qall")


class Sublime(Task):
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]

    def run_osx(self):
        Homebrew().cask_install("sublime-text")

        if os.path.isdir(os.path.join(PRIVATE, "subl")):
            symlink("Library/Application Support/Sublime Text 3", "~/.subl")
            symlink(os.path.join(PRIVATE, "subl", "User"), "~/.subl/Packages/User")
            symlink(os.path.join(PRIVATE, "subl", "INI"), "~/.subl/Packages/INI")

            # subl
            symlink("/Applications/Sublime Text.app/Contents/SharedSupport/bin/subl",
                    "/usr/local/bin/subl", sudo=True)

    def run(self):
        shell('sudo ln -sf "{df}/subl/rsub" /usr/local/bin'.format(df=DOTFILES))


class Ssmtp(Task):
    __platforms__ = ['ubuntu']

    def run_ubuntu(self):
        Apt().install("ssmtp")

        if os.path.isdir(os.path.join(PRIVATE, "ssmtp")):
            symlink(os.path.join(PRIVATE, "ssmtp", "ssmtp.conf"),
                    "/etc/ssmtp/ssmtp.conf", sudo=True)


class MySQL(Task):
    __platforms__ = ['linux', 'osx']

    def run(self):
        if os.path.isdir(os.path.join(PRIVATE, "mysql")):
            symlink(os.path.join(PRIVATE, "mysql", ".my.cnf"), "~/.my.cnf")


class OmniFocus(Task):
    __platforms__ = ['linux', 'osx']

    def run(self):
        shell('sudo ln -sf "{df}/omnifocus/omni" /usr/local/bin'.format(df=DOTFILES))


class LaTeX(Task):
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]

    def run_osx(self):
        Homebrew().cask_install("mactex")
        self.run()

    def run(self):
        if which("pdflatex"):
            mkdir("~/.local/bin")
            symlink(os.path.join(DOTFILES, "tex", "autotex"), "~/.local/bin/autotex")
            symlink(os.path.join(DOTFILES, "tex", "cleanbib"), "~/.local/bin/cleanbib")


class MacOSConfig(Task):
    """ macOS specific stuff """
    HUSHLOGIN = os.path.expanduser("~/.hushlogin")

    __platforms__ = ["osx"]
    __genfiles__ = [HUSHLOGIN]

    def run_osx(self):
        # disable "Last Login ..." messages on terminal
        if not os.path.exists(self.HUSHLOGIN):
            shell("touch " + self.HUSHLOGIN)


class MacOSApps(Task):
    CASKS = [
        'alfred',
        'anki',
        'bartender',
        'caffeine',
        'cmake',
        'disk-inventory-x',
        'fantastical',
        'flux',
        'google-drive',
        'google-earth-pro',
        'google-nik-collection',
        'google-photos-backup-and-sync',
        'istat-menus',
        'iterm2',
        'mendeley-desktop',
        'omnifocus',
        'omnigraffle',
        'omnioutliner',
        'omnipresence',
        'plex-media-player',
        'steam',
        'sublime-text',
        'texstudio',
        'transmission',
        'tunnelblick',
        'vlc',
    ]

    __platforms__ = ['osx']
    __deps__ = [Homebrew]

    def run(self):
        for cask in self.CASKS:
            Homebrew().cask_install(cask)


class GpuStat(Task):
    VERSION = "0.3.1"

    __platforms__ = ['linux', 'osx']
    __deps__ = [Python]

    def run(self):
        if which("nvidia-smi"):
            Python().pip_install("gpustat", self.VERSION)


class IOTop(Task):
    __platforms__ = ['ubuntu']

    def run_ubuntu(self):
        Apt().install("iotop")


class Ncdu(Task):
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]

    def run_osx(self):
        Homebrew().install("ncdu")

    def run_ubuntu(self):
        Apt().install("ncdu")


class HTop(Task):
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]

    def run_osx(self):
        Homebrew().install("htop")

    def run_ubuntu(self):
        Apt().install("htop")


class Emu(Task):
    VERSION = "0.2.5"

    __platforms__ = ['linux', 'osx']
    __deps__ = [Python]

    def run(self):
        Python().pip_install("emu", self.VERSION, sudo=True)


class JsonUtil(Task):
    JSONLINT_VERSION = "1.6.2"

    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]

    def run_osx(self):
        Homebrew().install("jq")
        self._run_common()

    def run_ubuntu(self):
        Apt().install("jq")
        self._run_common()

    def _run_common(self):
        Node().npm_install("jsonlint", self.JSONLINT_VERSION)


class Scripts(Task):
    __platforms__ = ['linux', 'osx']

    def run(self):
        symlink(os.path.join(DOTFILES, "media", "mkepisodal.py"),
               "~/.local/bin/mkepisodal")

        if HOSTNAME in ["florence", "diana", "ryangosling", "mary", "plod"]:
            symlink(os.path.join(DOTFILES, "servers", "mary"), "~/.local/bin/mary")
            symlink(os.path.join(DOTFILES, "servers", "diana"), "~/.local/bin/diana")

        if HOSTNAME in ["florence", "diana"]:
            symlink(os.path.join(DOTFILES, "servers", "ryan_gosling_have_my_photos.sh"),
                    "~/.local/bin/ryan_gosling_have_my_photos")
            symlink(os.path.join(DOTFILES, "servers", "ryan_gosling_have_my_music.sh"),
                    "~/.local/bin/ryan_gosling_have_my_music")
