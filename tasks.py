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
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]

    __genfiles__ = ["~/.local/bin/dropbox-find-conflicts"]

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
        if not which("dropbox"):
            raise OSError("dropbox must be installed manually")
        self._run_common()


class FluidApps(Task):
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
    __platforms__ = ['linux', 'osx']
    __osx_deps__ = [Homebrew]

    def run_osx(self):
        Homebrew().install("node")

    def run_ubuntu(self):
        Apt().install("nodejs")
        Apt().install("npm")
        symlink("/usr/bin/nodejs", "/usr/bin/node", sudo=True)


class ZSH(Task):
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
