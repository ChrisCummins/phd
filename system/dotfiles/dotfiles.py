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
# pylint: disable=missing-docstring
# pylint: disable=no-self-use
# pylint: disable=unused-wildcard-import
import json

from util import *


class Apt(object):
  """ debian package manager """

  def install_package(self, package):
    """ install a package using apt-get, return True if installed """
    if not shell_ok("/usr/bin/dpkg -s '{package}'".format(**vars())):
      shell("sudo apt-get install -y '{package}'".format(**vars()))
      return True

  def update(self):
    """ update package information """
    shell("sudo apt-get update")


class Sudo(Task):
  __platforms__ = ['linux']
  __deps__ = []
  __genfiles__ = ['/usr/bin/sudo']

  def install_linux(self):
    Apt().install_package('sudo')


class BuildEssential(Task):
  __platforms__ = ['linux']
  __deps__ = []
  __genfiles__ = ['/usr/bin/gcc']

  def install_linux(self):
    Apt().install_package('build-essential')


class Homebrew(Task):
  """ homebrew package manager """
  # Temporary files for caching list of installed packages and casks
  PKG_LIST = "/tmp/dotfiles_brew_package_list.txt"
  CASK_LIST = "/tmp/dotfiles_brew_cask_list.txt"
  OUTDATED_PKG_LIST = "/tmp/dotfiles_brew_outdated_packages.txt"
  OUTDATED_CASK_LIST = "/tmp/dotfiles_brew_outdated_casks.txt"

  BREW_BINARY = {
      'osx': '/usr/local/bin/brew',
      'ubuntu': '/home/linuxbrew/.linuxbrew/bin/brew',
  }[get_platform()]

  __platforms__ = ['linux', 'osx']
  __deps__ = []
  __ubuntu_deps__ = ['Sudo', 'BuildEssential']
  __genfiles__ = [BREW_BINARY]
  __tmpfiles__ = [PKG_LIST, CASK_LIST, OUTDATED_PKG_LIST, OUTDATED_CASK_LIST]

  def install_osx(self):
    if not which('brew'):
      task_print("Installing Homebrew")
      url = 'https://raw.githubusercontent.com/Homebrew/install/master/install'
      shell('yes | /usr/bin/ruby -e "$(curl -fsSL {url})"'.format(**vars()))
      shell('brew doctor')

  def install_ubuntu(self):
    # Install build dependencies:
    Apt().install_package("curl")
    Apt().install_package("file")
    Apt().install_package("git")
    Apt().install_package("python-setuptools")

    # On Ubuntu we create a special 'linuxbrew' user to own the linuxbrew
    # installation. This is because Homebrew does not support use by root user,
    # and on linux we can't guarantee that a user account exists, such as
    # in Docker containers.
    if not shell_ok('id -u linuxbrew'):
      shell('sudo useradd -m linuxbrew')

    if not os.path.exists('/home/linuxbrew/.linuxbrew/bin/brew'):
      url = ("https://raw.githubusercontent.com/"
             "Linuxbrew/install/master/install.sh")
      self.shell('yes | sh -c "$(curl -fsSL {url})"'.format(url=url))
      try:
        self.shell('{brew} doctor'.format(brew=self.BREW_BINARY))
      except CalledProcessError:
        # We don't care about errors at this stage.
        pass
      self._make_user_writeable()

  def _make_user_writeable(self):
    """Hacky workaround for non-user owned homebrew installations.

    Ensures that the directories (but not specific files) are writeable by
    all users on linux. This is so that programs which create files (such as
    python's pip) can write to their installation directories.
    """
    if PLATFORM in LINUX_DISTROS:
      shell(
          'find /home/linuxbrew/.linuxbrew/ -type d | xargs -L 500 sudo chmod 777'
      )

  @staticmethod
  def _as_linuxbrew_user(cmd):
    """Run a command as the 'linuxbrew' user."""
    return "sudo -H -u linuxbrew bash -c '{cmd}'".format(cmd=cmd)

  @classmethod
  def shell(cls, cmd):
    """Portability wrapper for the shell() command.

    On Linux, the command is executed as the linuxbrew user.
    """
    if PLATFORM in LINUX_DISTROS:
      return shell(cls._as_linuxbrew_user(cmd))
    else:
      return shell(cmd)

  @classmethod
  def shell_ok(cls, cmd):
    """Portability wrapper for the shell() command.

    On Linux, the command is executed as the linuxbrew user.
    """
    if PLATFORM in LINUX_DISTROS:
      return shell_ok(cls._as_linuxbrew_user(cmd))
    else:
      return shell_ok(cmd)

  @classmethod
  def brew_command(cls, cmd):
    cls.shell(" ".join([cls.BREW_BINARY, cmd]))

  def package_is_installed(self, package):
    """ return True if package is installed """
    if not os.path.isfile(self.PKG_LIST):
      self.shell("{self.BREW_BINARY} list > {self.PKG_LIST}".format(**vars()))

    return self.shell_ok("grep '^{package}$' <{self.PKG_LIST}".format(**vars()))

  def install_package(self, package, *opts):
    """ install a package using homebrew, return True if installed """
    if not self.package_is_installed(package):
      task_print("brew install " + package + ' ' + ' '.join(opts))
      self.shell("{self.BREW_BINARY} install {package}".format(**vars()))
      self._make_user_writeable()
      return True

  def package_is_outdated(self, package):
    """ returns True if package is outdated """
    if not self.package_is_installed(package):
      raise InvalidTaskError("homebrew package '{package}' cannot be upgraded "
                             "as it is not installed".format(**vars()))

    if not os.path.isfile(self.OUTDATED_PKG_LIST):
      self.shell(
          "{self.BREW_BINARY} outdated | awk '{{print $1}}' >{self.OUTDATED_PKG_LIST}"
          .format(**vars()))

    package_stump = package.split('/')[-1]
    return self.shell_ok(
        "grep '^{package_stump}$' <{self.OUTDATED_PKG_LIST}".format(**vars()))

  def upgrade_package(self, package):
    """ upgrade package, return True if upgraded """
    if self.package_is_outdated(package):
      task_print("brew upgrade {package}".format(**vars()))
      self.shell("{self.BREW_BINARY} upgrade {package}".format(**vars()))
      self._make_user_writeable()
      return True

  def cask_is_installed(self, cask):
    """ return True if cask is installed """
    if not os.path.isfile(self.CASK_LIST):
      self.shell(
          "{self.BREW_BINARY} cask list > {self.CASK_LIST}".format(**vars()))

    cask_stump = cask.split('/')[-1]
    return self.shell_ok(
        "grep '^{cask_stump}$' <{self.CASK_LIST}".format(**vars()))

  def install_cask(self, cask):
    """ install a homebrew cask, return True if installed """
    if not self.cask_is_installed(cask):
      task_print("brew cask install " + cask)
      self.shell("{self.BREW_BINARY} cask install {cask}".format(**vars()))
      self._make_user_writeable()
      return True

  def cask_is_outdated(self, cask):
    """ returns True if cask is outdated """
    if not self.cask_is_installed(cask):
      raise InvalidTaskError(
          "homebrew cask '{package}' cannot be upgraded as it is not installed".
          format(**vars()))

    if not os.path.isfile(self.OUTDATED_CASK_LIST):
      self.shell("{self.BREW_BINARY} cask outdated ".format(**vars()) +
                 "| awk '{{print $1}}' >{self.OUTDATED_CASK_LIST}".format(
                     **vars()))

    cask_stump = cask.split('/')[-1]
    return self.shell_ok(
        "grep '^{cask_stump}$' <{self.OUTDATED_CASK_LIST}".format(**vars()))

  def upgrade_cask(self, cask):
    """ upgrade a homebrew cask. does nothing if cask not installed """
    if self.cask_is_outdated(cask):
      task_print("brew cask upgrade {cask}".format(**vars()))
      self.shell("{self.BREW_BINARY} cask upgrade {cask}".format(**vars()))
      self._make_user_writeable()
      return True

  def uninstall_cask(self, cask):
    """ remove a homebrew cask, return True if uninstalled """
    if self.cask_is_installed(cask):
      task_print("brew cask remove " + cask)
      self.shell("{self.BREW_BINARY} cask remove " + cask)
      return True

  @staticmethod
  def _home():
    if get_platform() == 'osx':
      return '/usr/local'
    else:
      return '/home/linuxbrew/.linuxbrew'

  @classmethod
  def bin(cls, name):
    home = cls._home()
    return '{home}/bin/{name}'.format(**vars())

  @classmethod
  def lib(cls, name):
    home = cls._home()
    return '{home}/lib/{name}'.format(**vars())


class Python(Task):
  """ python 2 & 3 """
  PIP_LIST = "/tmp/dotfiles_pip_freeze.json"
  PYTHON2_BINARY = ('/usr/local/opt/python@2/bin/python2'
                    if PLATFORM == 'osx' else Homebrew.bin('python2'))
  PYTHON3_BINARY = ('/usr/local/opt/python@3/bin/python3'
                    if PLATFORM == 'osx' else Homebrew.bin('python3'))

  __platforms__ = ['linux', 'osx']
  __deps__ = ['Homebrew']
  __genfiles__ = []
  __genfiles__ = [
      PYTHON2_BINARY,
      PYTHON3_BINARY,
      Homebrew.bin('virtualenv'),
  ]
  __tmpfiles__ = [PIP_LIST]
  __versions__ = {
      "pip": "10.0.1",
      "virtualenv": "15.1.0",
  }

  def install(self):
    if Homebrew().install_package("python@2"):
      Homebrew.brew_command("link python@2 --force")

    if Homebrew().install_package("python"):
      Homebrew.brew_command("link python --force")

    # install pip
    self._install_pip_version(self.PYTHON2_BINARY, self.__versions__["pip"])
    self._install_pip_version(self.PYTHON3_BINARY, self.__versions__["pip"])

    # install virtualenv
    self.pip_install("virtualenv", self.__versions__["virtualenv"])

    # Symlink my preferred python into ~/.local/bin.
    shell('mkdir -p ~/.local/bin')
    symlink(self.PYTHON3_BINARY, "~/.local/bin/python")

  def _install_pip_version(self, python, version):
    if not shell_ok(
        "test $({python} -m pip --version | awk '{{print $2}}') = {version}".
        format(**vars())):
      task_print(
          "{python} -m pip install --upgrade 'pip=={version}'".format(**vars()))
      Homebrew.shell(
          '{python} -m pip install --upgrade "pip=={version}"'.format(**vars()))

  def upgrade(self):
    Homebrew().upgrade_package("python")
    Homebrew().upgrade_package("python3")

  def pip_install(self, package, version, python=PYTHON3_BINARY, sudo=False):
    """ install a package using pip, return True if installed """
    # Create the list of pip packages
    if os.path.exists(self.PIP_LIST):
      with open(self.PIP_LIST) as infile:
        data = json.loads(infile.read())
    else:
      data = {}

    if python not in data:
      freeze = Homebrew.shell(
          "{python} -m pip freeze 2>/dev/null".format(**vars()))
      data[python] = freeze.strip().split("\n")
      with open(self.PIP_LIST, "w") as outfile:
        json.dump(data, outfile)

    pkg_str = package + '==' + version
    if pkg_str not in data[python]:
      task_print(
          "{python} -m pip install {package}=={version}".format(**vars()))
      Homebrew.shell(
          "{python} -m pip install {package}=={version}".format(**vars()))
      return True


class PypiConfig(Task):
  """pypi config file"""
  PYP_IRC = "~/.pypirc"

  __platforms__ = ['linux', 'osx']
  __deps__ = ["Python"]
  __reqs__ = [lambda: os.path.isdir(PRIVATE + "/python/")]
  __genfiles__ = [PYP_IRC]

  def install(self):
    symlink("{private}/python/.pypirc".format(private=PRIVATE), "~/.pypirc")


class Docker(Task):
  __platforms__ = ['osx', 'linux']
  __deps__ = ['Homebrew']

  def install_osx(self):
    Homebrew().install_cask('docker')

  def install_ubuntu(self):
    Apt().install_package('docker.io')
    shell('sudo systemctl start docker')
    shell('sudo systemctl enable docker')
    shell('sudo usermod -aG docker $USER')

  def uninstall(self):
    Homebrew().uninstall_cask('docker')


class Unzip(Task):
  """ unzip pacakge """
  __platforms__ = ['osx', 'ubuntu']
  __deps__ = ['Homebrew']
  __genfiles__ = [Homebrew.bin('unzip')]

  def install(self):
    Homebrew().install_package("unzip")
    brew = Homebrew.BREW_BINARY
    shell("{brew} link unzip --force".format(**vars()))

  def upgrade(self):
    Homebrew().upgrade_package("unzip")


class Ruby(Task):
  """ ruby environment """
  __platforms__ = ['osx']
  __osx_deps__ = ['Homebrew']
  __genfiles__ = ['~/.rbenv']
  __versions__ = {"ruby": "2.4.1"}

  def install_osx(self):
    Homebrew().install_package("rbenv")

    # initialize rbenv if required
    if shell_ok("which rbenv"):
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
  __deps__ = ['Homebrew']
  __genfiles__ = [Homebrew.bin('curl')]

  def install(self):
    Homebrew().install_package("curl")
    brew = Homebrew.BREW_BINARY
    shell("{brew} link curl --force".format(**vars()))

  def upgrade(self):
    Homebrew().upgrade_package("curl")


class DropboxInbox(Task):
  """ dropbox inbox """
  __platforms__ = ['linux', 'osx']
  __deps__ = ['Dropbox']
  __reqs__ = [lambda: os.path.isdir(os.path.expanduser("~/Dropbox/Inbox"))]
  __genfiles__ = ["~/Inbox"]

  def install(self):
    if not os.path.isdir(os.path.expanduser("~/Inbox")):
      symlink("Dropbox/Inbox", "~/Inbox")


class DropboxScripts(Task):
  """ dropbox scripts """
  __platforms__ = ['linux', 'osx']
  __deps__ = ['Dropbox']
  __reqs__ = [lambda: os.path.isdir(os.path.expanduser("~/Dropbox"))]
  __genfiles__ = ["~/.local/bin/dropbox-find-conflicts"]

  def install(self):
    symlink(
        usr_share("Dropbox/dropbox-find-conflicts.sh"),
        "~/.local/bin/dropbox-find-conflicts")


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

  def install_osx(self):
    Homebrew().install_cask("dropbox")
    self._install_common()

  def install_linux(self):
    if (not os.path.exists(os.path.expanduser("~/.dropbox-dist/dropboxd")) and
        not IS_TRAVIS_CI):  # skip on Travis CI:
      task_print("Installing Dropbox")
      shell(
          'cd - && wget -O - "{self.UBUNTU_URL}" | tar xzf -'.format(**vars()))
      self.installed_on_ubuntu = True
    self._install_common()

  def upgrade_osx(self):
    Homebrew().upgrade_cask("dropbox")

  def teardown(self):
    if self.installed_on_ubuntu:
      logging.info("")
      logging.info(
          "NOTE: manual step required to complete dropbox installation:")
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
            shell("cp -r '{}/fluid.apps/{}' '/Applications/{}'".format(
                PRIVATE, app, app))

  def upgrade_osx(self):
    Homebrew().upgrade_cask("fluid")


class SshConfig(Task):
  """ ssh configuration """
  __platforms__ = ['linux', 'osx']
  __reqs__ = [lambda: os.path.isdir(PRIVATE + "/ssh")]
  __genfiles__ = [
      "~/.ssh/authorized_keys",
      "~/.ssh/known_hosts",
      "~/.ssh/config",
  ]

  def install(self):
    # Dropbox doesn't sync file permissions. Restore them here.
    shell('find {}/ssh -type f -exec chmod 0600 {{}} \;'.format(PRIVATE))

    mkdir("~/.ssh")
    for file in ['authorized_keys', 'known_hosts', 'config']:
      src = os.path.join(PRIVATE, "ssh", file)
      dst = os.path.join("~/.ssh", file)
      if shell_ok("test $(stat -c %U '{src}') = $USER".format(**vars())):
        symlink(src, dst)
      else:
        copy_file(src, dst)

    host_dir = os.path.join(PRIVATE, 'ssh', HOSTNAME)
    if os.path.isdir(host_dir):
      copy_file(os.path.join(host_dir, "id_rsa"), "~/.ssh/id_rsa")
      copy_file(os.path.join(host_dir, "id_rsa.pub"), "~/.ssh/id_rsa.pub")

    shell("chmod 600 ~/.ssh/*")


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
      logging.info(
          "NOTE: manual steps required to complete netdata installation:")
      logging.info("    $ " + Colors.BOLD + Colors.RED + "crontab -e" +
                   Colors.END)
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
      logging.info(
          "NOTE: manual steps required to complete Wacom driver setup:")
      logging.info(
          "    " + Colors.BOLD + Colors.RED +
          "Enable Wacom kernel extension in System Preferences > Security & Privacy"
          + Colors.END)


class Node(Task):
  """ nodejs and npm """
  PKG_LIST = "/tmp/npm-list.txt"
  NPM_BINARY = Homebrew.bin('npm')
  NODE_BINARY = Homebrew.bin('node')

  __platforms__ = ['linux', 'osx']
  __deps__ = ['Homebrew']
  __genfiles__ = [
      NPM_BINARY,
      NODE_BINARY,
  ]
  __tmpfiles__ = [PKG_LIST]

  def install(self):
    Homebrew().install_package("node")

  def upgrade(self):
    Homebrew().upgrade_package("node")

  def npm_install(self, package, version):
    """ install a package using npm, return True if installed """
    # Create the list of npm packages
    if not os.path.isfile(self.PKG_LIST):
      shell("{self.NPM_BINARY} list -g > {self.PKG_LIST}".format(**vars()))

    if not shell_ok(
        "grep '{package}@{version}' <{self.PKG_LIST}".format(**vars())):
      task_print("npm install -g {package}@{version}".format(**vars()))
      shell("{self.NPM_BINARY} install -g {package}@{version}".format(**vars()))
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
      '~/.zshenv',
      '~/.zshrc',
  ]
  __osx_genfiles__ = ['/usr/local/bin/zsh']
  __linux_genfiles__ = ['/bin/zsh']
  __versions__ = {
      "oh-my-zsh": "c3b072eace1ce19a48e36c2ead5932ae2d2e06d9",
      "zsh-syntax-highlighting": "b07ada1255b74c25fbc96901f2b77dc4bd81de1a",
  }

  def install_osx(self):
    Homebrew().install_package("zsh")
    self.install_common()

  def install_linux(self):
    Apt().install_package('zsh')
    self.install_common()

  def install_common(self):
    # install config files
    symlink(usr_share("Zsh"), "~/.zsh")
    symlink(usr_share("Zsh/zshrc"), "~/.zshrc")
    symlink(usr_share("Zsh/zshenv"), "~/.zshenv")

    # oh-my-zsh
    clone_git_repo(
        github_repo("robbyrussell", "oh-my-zsh"), "~/.oh-my-zsh",
        self.__versions__["oh-my-zsh"])
    symlink("~/.zsh/cec.zsh-theme", "~/.oh-my-zsh/custom/cec.zsh-theme")

    # syntax highlighting module
    clone_git_repo(
        github_repo("zsh-users", "zsh-syntax-highlighting"),
        "~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting",
        self.__versions__["zsh-syntax-highlighting"])

  def upgrade(self):
    Homebrew().upgrade_package("zsh")


class ZshPrivate(Task):
  """ zsh private config files """
  __platforms__ = ['linux', 'osx']
  __deps__ = ['Zsh']
  __reqs__ = [lambda: os.path.isdir(os.path.join(PRIVATE, "zsh"))]
  __genfiles__ = ["~/.zsh/private"]

  def install(self):
    symlink(os.path.join(PRIVATE, "zsh"), "~/.zsh/private")


class ZshBazelCompletion(Task):
  """ tab completion for bazel """
  __platforms__ = ['linux', 'osx']
  __deps__ = ['Zsh']
  __genfiles__ = ['~/.zsh/completion/_bazel', '~/.zsh/cache']
  __versions__ = {
      "bazel_completion": "30af177d5cd2188ee6e23ba849d865b8a42ad8f8",
  }

  def install(self):
    url = ('https://raw.githubusercontent.com/bazelbuild/bazel/{}/'
           'scripts/zsh_completion/_bazel').format(
               self.__versions__['bazel_completion'])
    bazel = os.path.expanduser("~/.bazel_tmp")
    shell("rm -rf {bazel}".format(**vars()))

    if not os.path.isfile(os.path.expanduser("~/.zsh/completion/_bazel")):
      # FIXME: shell("fpath[1,0]=~/.zsh/completion/")
      mkdir("~/.zsh/completion/")
      shell("wget {url} -O ~/.zsh/completion/_bazel".format(**vars()))

    if not os.path.isdir("~/.zsh/cache"):
      mkdir("~/.zsh/cache")


class Autoenv(Task):
  """ 'cd' wrapper """
  __platforms__ = ['linux', 'osx']
  __deps__ = ['Python']
  __genfiles__ = [Homebrew.bin('activate.sh')]

  def install(self):
    Python().pip_install("autoenv", "1.0.0")


class Lmk(Task):
  """ let-me-know """
  __platforms__ = ['linux', 'osx']
  __deps__ = ['Python', 'Phd']
  __genfiles__ = ['/usr/local/bin/lmk']

  def install(self):
    symlink('~/phd/util/lmk/lmk.py', '/usr/local/bin/lmk', sudo=True)


class LmkConfig(Task):
  """ let-me-know config file """
  __platforms__ = ['linux', 'osx']
  __deps__ = ['Lmk']
  __reqs__ = [lambda: os.path.isdir(os.path.join(PRIVATE, "lmk"))]
  __genfiles__ = ["~/.lmkrc"]

  def install(self):
    symlink(os.path.join(PRIVATE, "lmk", "lmkrc"), "~/.lmkrc")


class Git(Task):
  """ git config """
  __platforms__ = ['linux', 'osx']
  __osx_deps__ = ['Homebrew']
  __genfiles__ = ['~/.gitconfig']

  def install(self):
    Homebrew().install_package('git')
    if not IS_TRAVIS_CI:
      symlink(usr_share("git/gitconfig"), "~/.gitconfig")

  def upgrade(self):
    Homebrew().upgrade_package("git")


class GitLfs(Task):
  """git-lfs"""
  __platforms__ = ['linux', 'osx']
  __deps__ = ['Homebrew']
  __genfiles__ = [Homebrew.bin('git-lfs')]

  def install(selfs):
    Homebrew().install_package('git-lfs')

  def upgrade(self):
    Homebrew().upgrade_package('git-lfs')


class GitPrivate(Task):
  """ git private config """
  __platforms__ = ['linux', 'osx']
  __osx_deps__ = ['Homebrew']
  __reqs__ = [lambda: os.path.isdir(os.path.join(PRIVATE, "git"))]
  __genfiles__ = [
      '~/.githubrc',
      '~/.gogsrc',
  ]

  def install(self):
    symlink(os.path.join(PRIVATE, "git", "githubrc"), "~/.githubrc")
    symlink(os.path.join(PRIVATE, "git", "gogsrc"), "~/.gogsrc")

  def upgrade_osx(self):
    Homebrew().upgrade_package("git")


class Gogs(Task):
  """self hosted git client"""
  __platforms__ = ["linux"]
  __hosts__ = ["ryangosling"]
  __deps__ = ["MySQL"]
  __genfiles__ = [
      "/opt/gogs",
      '/var/log/gogs.log',
  ]
  __versions__ = {
      "gogs": "0.11.34",
  }

  def install(self):
    url = ("https://dl.gogs.io/{self.__versions__['gogs']}/linux_amd64.tar.gz".
           format(**vars()))
    shell("wget {url} -O /tmp/gogs.tar.gz".format(**vars()))
    shell("cd /tmp && tar xjf linux_amd64.tar.gz")
    shell("rm linux_amd64.tar.gz")
    shell("sudo mv gogs /opt/gogs")
    shell("sudo chown -R cec:cec /opt/gogs")
    if not os.path.isfile('/var/log/gogs.log'):
      shell("sudo ln -s /opt/gogs/log/gogs.log /var/log/gogs.log")


class GogsConfig(Task):
  """custom config for gogs"""
  __platforms__ = ["linux"]
  __hosts__ = ["ryangosling"]
  __deps__ = ["Gogs", "MySQL"]
  __genfiles__ = ["/opt/gogs/custom/conf/app.ini"]

  def install(self):
    symlink(
        usr_share("gogs/custom/conf/app.ini"), "/opt/gogs/custom/conf/app.ini")


class Wallpaper(Task):
  """ set desktop background """
  WALLPAPERS = {
      "diana": "~/Dropbox/Pictures/desktops/diana/Manhattan.jpg",
      "florence": "~/Dropbox/Pictures/desktops/florence/Uluru.jpg",
  }
  __hosts__ = WALLPAPERS.keys()
  __platforms__ = ['osx']

  def install_osx(self):
    path = os.path.expanduser(self.WALLPAPERS[HOSTNAME])
    if os.path.exists(path):
      shell("osascript -e 'tell application \"Finder\" to set " +
            "desktop picture to POSIX file \"{path}\"'".format(**vars()))


class GnuCoreutils(Task):
  """ replace BSD utils with GNU """
  __platforms__ = ['linux', 'osx']
  __osx_deps__ = ['Homebrew']
  __osx_genfiles__ = [
      '/usr/local/opt/coreutils/libexec/gnubin/cp',
      '/usr/local/opt/gnu-sed/libexec/gnubin/sed',
      '/usr/local/opt/gnu-tar/libexec/gnubin/tar',
  ]

  def install_linux(self):
    # Already there.
    pass

  def install_osx(self):
    Homebrew().install_package('coreutils')
    Homebrew().install_package('findutils')
    Homebrew().install_package('gnu-indent')
    Homebrew().install_package('gnu-sed')
    Homebrew().install_package('gnu-tar')
    Homebrew().install_package('gnu-time')
    Homebrew().install_package('gnu-which')

  def upgrade_osx(self):
    Homebrew().upgrade_package("coreutils")
    Homebrew().upgrade_package('findutils')
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
  __genfiles__ = [Homebrew.bin('diff-so-fancy')]

  def install(self):
    Node().npm_install("diff-so-fancy", self.VERSION)


class Nbdime(Task):
  """ diffs for Jupyter notebooks """
  VERSION = '1.0.0'

  __platforms__ = ['linux', 'osx']
  __deps__ = ['Git', 'Python']
  __genfiles__ = [Homebrew.bin('nbdime')]

  def install(self):
    Python().pip_install('nbdime', self.VERSION)


class GhArchiver(Task):
  """ github archiver """
  VERSION = "0.0.6"

  __platforms__ = ['linux', 'osx']
  __deps__ = ['Python']
  __genfiles__ = [Homebrew.bin('gh-archiver')]

  def install(self):
    Python().pip_install("gh-archiver", self.VERSION)


class Tmux(Task):
  """ tmux config """
  __platforms__ = ['linux', 'osx']
  __genfiles__ = ['~/.tmux.conf']
  __osx_genfiles__ = ['/usr/local/bin/tmux']
  __linux_genfiles__ = ['/usr/bin/tmux']

  def install_osx(self):
    Homebrew().install_package("tmux")
    self._install_common()

  def install_linux(self):
    Apt().install_package("tmux")
    self._install_common()

  def _install_common(self):
    symlink(usr_share("tmux/tmux.conf"), "~/.tmux.conf")

  def upgrade_osx(self):
    Homebrew().upgrade_package("tmux")

  def upgrade_linux(self):
    Apt().upgrade_package("tmux")


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
    self.install_common()

  def install_linux(self):
    Apt().install_package('vim')
    self.install_common()

  def install_common(self):
    symlink(usr_share("Vim/vimrc"), "~/.vimrc")

    # Vundle
    clone_git_repo(
        github_repo("VundleVim", "Vundle.vim"), "~/.vim/bundle/Vundle.vim",
        self.__versions__["vundle"])
    if os.path.isfile(Homebrew.bin('vim')):
      # We use the absolute path to vim since on first run we won't
      # necessarily have the homebrew bin directory in out $PATH.
      vim = Homebrew.bin('vim')
    else:
      vim = 'vim'
    shell("{vim} +PluginInstall +qall".format(vim=vim))

  def upgrade_osx(self):
    Homebrew().upgrade_package("vim")


class Linters(Task):
  __platforms__ = ['osx']
  __osx_deps__ = ['Node', 'Python', 'Go']
  __osx_genfiles__ = [
      '/usr/local/bin/cpplint',
      '/usr/local/bin/csslint',
      '/usr/local/bin/pycodestyle',
      '~/.config/pycodestyle',
      '~/go/bin/protoc-gen-lint',
      '/usr/local/bin/write-good',
      '/usr/local/bin/writegood',
      Homebrew.bin("tidy"),
      Homebrew.bin("buildifier"),
  ]
  __versions__ = {
      "cpplint": "1.3.0",
      "csslint": "1.0.5",
      "pycodestyle": "2.3.1",
      "write-good": "0.11.3",
  }

  def install_osx(self):
    Python().pip_install("cpplint", version=self.__versions__["cpplint"])
    Node().npm_install("csslint", version=self.__versions__["csslint"])
    Python().pip_install(
        "pycodestyle",
        version=self.__versions__["pycodestyle"],
        python=Python.PYTHON3_BINARY)
    Homebrew().install_package("tidy-html5")
    Homebrew().install_package("buildifier")
    Go().get('github.com/ckaznocha/protoc-gen-lint')
    Node().npm_install("write-good", version=self.__versions__["write-good"])

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
    symlink(
        "/Applications/Sublime Text.app/Contents/SharedSupport/bin/subl",
        "/usr/local/bin/subl",
        sudo=True)

    symlink("~/Library/Application Support/Sublime Text 3", "~/.subl")

    self.install()

  def install(self):
    symlink(usr_share("Sublime Text/rsub"), "/usr/local/bin/rsub", sudo=True)

  def upgrade_osx(self):
    Homebrew().upgrade_cask("sublime-text")


class SublimeConfig(Task):
  """ sublime text """
  __platforms__ = ['osx']
  __osx_deps__ = ['SublimeText']
  __reqs__ = [lambda: os.path.isdir(os.path.join(PRIVATE, "subl"))]
  __genfiles__ = [
      "~/.subl",
      "~/.subl/Packages/User",
      "~/.subl/Packages/INI",
  ]

  def install_osx(self):
    symlink(os.path.join(PRIVATE, "subl", "User"), "~/.subl/Packages/User")
    symlink(os.path.join(PRIVATE, "subl", "INI"), "~/.subl/Packages/INI")


class JetbrainsIDEs(Task):
  # A map of homebrew cask names to local apps.
  IDES = {
      "datagrip": "/Applications/DataGrip.app",
      "goland": "/Applications/GoLand.app",
      "clion": "/Applications/CLion.app",
      "intellij-idea": "/Applications/IntelliJ IDEA.app",
      "pycharm": "/Applications/PyCharm.app",
  }

  __platforms__ = ['osx']
  __deps__ = ['Homebrew']
  __genfiles__ = IDES.values()

  def install(self):
    for ide in self.IDES:
      Homebrew().install_cask(ide)

  def upgrade(self):
    for ide in self.IDES:
      Homebrew().upgrade_cask(ide)


class Ssmtp(Task):
  """ mail server """
  __platforms__ = ['ubuntu']
  __genfiles__ = ["/usr/sbin/ssmtp"]

  def install_ubuntu(self):
    Apt().install_package("ssmtp")


class SsmtpConfig(Task):
  """ mail config """
  __platforms__ = ['ubuntu']
  __deps__ = ["Ssmtp"]
  __genfiles__ = ["/etc/ssmtp/ssmtp.conf"]

  def install_ubuntu(self):
    symlink(
        os.path.join(PRIVATE, "ssmtp", "ssmtp.conf"),
        "/etc/ssmtp/ssmtp.conf",
        sudo=True)


class MySQL(Task):
  """ mysql pacakge """
  __platforms__ = ['linux', 'osx']
  __osx_genfiles__ = [Homebrew.bin("mysql")]
  __linux_genfiles__ = ['/usr/bin/mysql']
  __deps__ = ['Homebrew']

  def install_osx(self):
    Homebrew().install_package("mysql")
    Homebrew.brew_command("services start mysql")

  def install_ubuntu(self):
    # Currently (2018-05-17), homebrew mysql does not appear to work.
    Apt().install_package("mysql-server")
    shell("sudo systemctl enable mysql")


class MySQLConfig(Task):
  """ mysql configuration """
  __platforms__ = ['linux', 'osx']
  __genfiles__ = ["~/.my.cnf"]
  __reqs__ = [lambda: os.path.isdir(os.path.join(PRIVATE, "mysql"))]

  def install(self):
    symlink(os.path.join(PRIVATE, "mysql", ".my.cnf"), "~/.my.cnf")


class LaTeX(Task):
  """ latex compiler and libraries """
  __platforms__ = ['linux', 'osx']
  __osx_deps__ = ['Homebrew']
  __osx_genfiles__ = [
      '/Library/TeX/Distributions/.DefaultTeX/Contents/Programs/texbin/pdflatex',
      '/Applications/texstudio.app',
      '/Applications/texstudio.app/Contents/Resources/en_GB.ign'
  ]

  def install_osx(self):
    Homebrew().install_cask("mactex")
    Homebrew().install_cask("texstudio")
    # The poppler package contains the tool pdffonts, useful detecting Type 3
    # fonts.
    Homebrew().install_package('poppler')
    self.install()
    symlink(os.path.join(PRIVATE, 'texstudio', 'en_GB.ign'),
            '/Applications/texstudio.app/Contents/Resources/en_GB.ign')

  def install_linux(self):
    Apt().install_package('texlive-full')
    Apt().install_package('biber')

  def upgrade_osx(self):
    Homebrew().upgrade_cask("mactex")
    Homebrew().upgrade_cask("texstudio")


class LaTeXScripts(Task):
  """ latex helper scripts """
  __platforms__ = ['linux', 'osx']
  __osx_deps__ = ['LaTeX']
  __reqs__ = [lambda: which("pdflatex")]
  __genfiles__ = ["~/.local/bin/autotex", "~/.local/bin/cleanbib"]

  def install(self):
    mkdir("~/.local/bin")
    symlink(usr_share("LaTeX", "autotex"), "~/.local/bin/autotex")
    symlink(usr_share("LaTeX", "cleanbib"), "~/.local/bin/cleanbib")


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
    if not os.path.exists(
        '/Applications/Adobe Lightroom CC/Adobe Lightroom CC.app'):
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
      logging.info(
          "NOTE: manual step required to complete creative cloud installation:")
      logging.info(
          "    $ " + Colors.BOLD + Colors.RED +
          "open '/usr/local/Caskroom/adobe-creative-cloud/latest/Creative Cloud Installer.app'"
          + Colors.END)
      logging.info(
          "    $ " + Colors.BOLD + Colors.RED +
          "open '/usr/local/Caskroom/google-nik-collection/1.2.11/Nik Collection.app'"
          + Colors.END)


class MacOSConfig(Task):
  """ macOS specific stuff """
  HUSHLOGIN = os.path.expanduser("~/.hushlogin")

  __platforms__ = ["osx"]
  __genfiles__ = ['~/.hushlogin']

  def add_login_item(self, path, hidden):
    """ add a program as a login item """
    path = os.path.abspath(path)
    hidden = "true" if hidden else "false"

    shell("osascript -e 'tell application \"System Events\" to make login "
          "item at end with properties {{path:\"{path}\", hidden:{hidden}}}'".
          format(**vars()))

  def install_osx(self):
    # Based on: https://github.com/holman/dotfiles/blob/master/macos/set-defaults.sh

    # Disable press-and-hold for keys in favor of key repeat.
    shell("defaults write -g ApplePressAndHoldEnabled -bool false")

    # Use AirDrop over every interface.
    shell("defaults write com.apple.NetworkBrowser BrowseAllInterfaces 1")

    # Always open everything in Finder's list view. This is important.
    shell("defaults write com.apple.Finder FXPreferredViewStyle Nlsv")

    # Show the ~/Library folder.
    shell("chflags nohidden ~/Library")

    # Set a really fast key repeat.
    shell("defaults write NSGlobalDomain KeyRepeat -int 1")

    # Set the Finder prefs for showing a few different volumes on the Desktop.
    shell(
        "defaults write com.apple.finder ShowExternalHardDrivesOnDesktop -bool false"
    )
    shell(
        "defaults write com.apple.finder ShowRemovableMediaOnDesktop -bool false"
    )

    # Run the screensaver if we're in the bottom-left hot corner.
    shell("defaults write com.apple.dock wvous-bl-corner -int 5")
    shell("defaults write com.apple.dock wvous-bl-modifier -int 0")

    # Set up Safari for development.
    shell("defaults write com.apple.Safari IncludeInternalDebugMenu -bool true")
    shell("defaults write com.apple.Safari IncludeDevelopMenu -bool true")
    shell(
        "defaults write com.apple.Safari WebKitDeveloperExtrasEnabledPreferenceKey -bool true"
    )
    shell(
        'defaults write com.apple.Safari "com.apple.Safari.ContentPageGroupIdentifier.WebKit2DeveloperExtrasEnabled" -bool true'
    )
    shell("defaults write NSGlobalDomain WebKitDeveloperExtras -bool true")

    # disable "Last Login ..." messages on terminal
    if not os.path.exists(os.path.expanduser("~/.hushlogin")):
      task_print("Creating ~/.hushlogin")
      shell("touch " + os.path.expanduser("~/.hushlogin"))


class Caffeine(Task):
  __platforms__ = ['osx']
  __deps__ = ['Homebrew', 'MacOSConfig']
  __osx_genfiles__ = ['/Applications/Caffeine.app']

  def install(self):
    Homebrew().install_cask('caffeine')
    # TODO(cec): Re-activate this once osascript command is fixed.
    # MacOSConfig().add_login_item('/Applications/Caffeine.app', hidden=False)

  def upgrade(self):
    Homebrew().upgrade_cask('caffeine')


class HomebrewCasks(Task):
  """ macOS homebrew binaries """
  CASKS = {
      'alfred': '/Applications/Alfred 3.app',
      'anki': '/Applications/Anki.app',
      'bartender': '/Applications/Bartender 3.app',
      'bettertouchtool': '/Applications/BetterTouchTool.app',
      'calibre': '/Applications/calibre.app',
      'dash': '/Applications/Dash.app',
      'disk-inventory-x': '/Applications/Disk Inventory X.app',
      'etcher': '/Applications/Etcher.app',
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
      # TODO(cec): Issue post-install message for Rocket telling the user to
      # launch the app, make the changes to Accessibility seettings, and hide
      # the menubar icon using Bartender.
      'rocket': '/Applications/Rocket.app',
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
  TRASH_BIN = Homebrew.bin('trash')

  __platforms__ = ['linux', 'osx']
  __deps__ = ['Node']
  __genfiles__ = [TRASH_BIN]
  __versions__ = {"trash-cli": "1.4.0"}

  def install(self):
    Node().npm_install('trash-cli', self.__versions__["trash-cli"])

  def trash(self, *paths):
    for path in paths:
      trash_bin = self.TRASH_BIN
      path = os.path.expanduser(path)
      shell("{trash_bin} '{path}'".format(**vars()))


class Emacs(Task):
  """ emacs text editor and config """
  __platforms__ = ['linux', 'osx']
  __deps__ = ['Trash']
  __osx_deps__ = ['Homebrew']
  __versions__ = {
      "prelude": "f7d5d68d432319bb66a5f9410d2e4eadd584f498",
  }

  def install(self):
    Homebrew().install_cask('emacs')
    self._install_common()

  def install_ubuntu(self):
    Homebrew().install_package('emacs')
    self._install_common()

  def _install_common(self):
    clone_git_repo(
        github_repo("bbatsov", "prelude"), "~/.emacs.d",
        self.__versions__["prelude"])
    # prelude requires there be no ~/.emacs file on first run
    Trash().trash('~/.emacs')

  def upgrade(self):
    Homebrew().upgrade_cask('emacs')


class Graphviz(Task):
  """ graph visualization software """
  __platforms__ = ['osx', 'linux']
  __deps__ = ['Homebrew']

  def install(self):
    Homebrew().install_package('graphviz')


class AppStore(Task):
  """ install mas app store command line """
  __platforms__ = ['osx']
  __deps__ = ['Homebrew']

  def install(self):
    Homebrew().install_package('mas')

    # Check that dotfiles was configured with Apple ID
    if not APPLE_ID:
      logging.critical(
          "\nerror: Apple ID not set! Run ./configure --apple-id <email>")
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
    if os.path.exists(package_dest):
      return False
    else:
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

  __platforms__ = ['osx']
  __deps__ = ['AppStore']
  __genfiles__ = list(APPS.values())

  def install(self):
    for app_id in self.APPS:
      AppStore().install_app(app_id, self.APPS[app_id])


class DianaApps(Task):
  APPS = {883878097: '/Applications/Server.app'}

  __platforms__ = ['osx']
  __hosts__ = ['diana']
  __genfiles__ = list(APPS.values())

  def install(self):
    for app_id in self.APPS:
      AppStore().install_app(app_id, self.APPS[app_id])


class GpuStat(Task):
  """ nice nvidia-smi wrapper """
  VERSION = "0.3.1"

  __platforms__ = ['linux', 'osx']
  __deps__ = ['Python']
  __reqs__ = [lambda: which("nvidia-smi")]
  __genfiles__ = [
      '/usr/local/bin/gpustat',
  ]

  def install(self):
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

  def install(self):
    Homebrew().install_package("ncdu")

  def upgrade(self):
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


class Go(Task):
  """ go compiler """
  __platforms__ = ['linux', 'osx']
  __deps__ = ['Homebrew']
  __genfiles__ = [Homebrew.bin("go")]

  def install(self):
    Homebrew().install_package('go')

  def upgrade(self):
    Homebrew().upgrade_package('go')

  def get(self, package):
    shell("cd ~ && go get {package}".format(package=package))


class OmniFocus(Task):
  """ task manager and utilities """
  __platforms__ = ['linux', 'osx']
  __osx_deps__ = ['Homebrew', 'Java']
  __genfiles__ = ["/usr/local/bin/omni"]
  __osx_genfiles__ = [
      "/Applications/OmniFocus.app",
      "/usr/local/opt/ofexport/bin/of2",
  ]
  __versions__ = {
      "ofexport": "1.0.20",
  }

  OFEXPORT_URL = "https://github.com/psidnell/ofexport2/archive/ofexport-v2-" + \
                 __versions__["ofexport"] + ".zip"

  def install_osx(self):
    Homebrew().install_cask('omnifocus')

    # Check that of2 is installed and is the correct version
    if (not os.path.exists("/usr/local/opt/ofexport/bin/of2") and
        shell("/usr/local/opt/ofexport/bin/of2 -h").split("\n")[2] !=
        "Version: " + self.__versions__["ofexport"]):
      task_print("Downloading ofexport")
      shell("rm -rf /usr/local/opt/ofexport")
      url, ver = self.OFEXPORT_URL, self.__versions__["ofexport"]
      shell("wget {url} -O /tmp/ofexport.zip".format(**vars()))
      task_print("Installing ofexport")
      shell("unzip -o /tmp/ofexport.zip")
      shell("rm -f /tmp/ofexport.zip")
      shell("mv ofexport2-ofexport-v2-{ver} /usr/local/opt/ofexport".format(
          **vars()))

    # Run common-install commands:
    self.install()

  def install(self):
    symlink(usr_share("OmniFocus/omni"), "/usr/local/bin/omni", sudo=True)

  def upgrade_osx(self):
    Homebrew().upgrade_cask('omnifocus')


class Timing(Task):
  """ time tracking app """
  __platforms__ = ['osx']
  __osx_genfiles__ = ['/Applications/Timing.app']
  __osx_deps__ = ['Homebrew']

  def install_osx(self):
    Homebrew().install_cask("timing")


class Timer(Task):
  """ time tracking CLI """
  __platforms__ = ['linux', 'osx']
  __genfiles__ = ['~/.local/bin/timer']

  def install(self):
    mkdir("~/.local/bin")
    symlink(usr_share("timer", "timer.py"), "~/.local/bin/timer")


class MeCsv(Task):
  """ me.csv health and time tracking """
  __platforms__ = ['osx']
  __osx_deps__ = ['OmniFocus']
  __reqs__ = [lambda: os.path.isdir(os.path.join(PRIVATE, "me.csv"))]
  __genfiles__ = ["~/.me.json", "~/me.csv"]

  def install_osx(self):
    symlink(os.path.join(PRIVATE, "me.csv", "config.json"), "~/.me.json")
    symlink(os.path.join(PRIVATE, "me.csv", "data"), "~/me.csv")


class Bazel(Task):
  """ bazel build system """
  __platforms__ = ['linux', 'osx']
  __deps__ = ['Curl']
  __osx_deps__ = ['Java', 'Homebrew']
  __linux_deps__ = ['Zip']
  __genfiles__ = ['/usr/local/bin/bazel']

  def install_osx(self):
    Homebrew().install_package('bazel')

  def install_ubuntu(self):
    # Currently (2018-05-17) I have been unable to get the Linuxbrew
    # distribution of Bazel to build.
    # See: https://docs.bazel.build/versions/master/install-ubuntu.html
    if not os.path.isfile('/usr/local/bin/bazel'):
      shell(
          'curl -L -o /tmp/bazel.sh https://github.com/bazelbuild/bazel/releases/download/0.14.1/bazel-0.14.1-installer-linux-x86_64.sh'
      )
      shell('sudo bash /tmp/bazel.sh')
      shell('rm /tmp/bazel.sh')

  def upgrade_osx(self):
    Homebrew().upgrade_package("bazel")


class Buildifier(Task):
  __platforms__ = ['linux', 'osx']
  __deps__ = ['Homebrew', 'Bazel']
  __genfiles__ = [Homebrew.bin('buildifier')]

  def install(self):
    Homebrew().install_package('buildifier')


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

  def install(self):
    Homebrew().install_package('protobuf')

  def upgrade(self):
    Homebrew().upgrade_package("protobuf")


class Sloccount(Task):
  """ source line count """
  __platforms__ = ['linux', 'osx']
  __osx_deps__ = ['Homebrew']
  __osx_genfiles__ = ['/usr/local/bin/sloccount']
  __linux_genfiles__ = ['/usr/bin/sloccount']

  def install(self):
    Homebrew().install_package('sloccount')

  def upgrade(self):
    Homebrew().upgrade_package("sloccount")


class Emu(Task):
  """ backup software """
  VERSION = "0.3.0"

  __platforms__ = ['linux', 'osx']
  __deps__ = ['Python']
  __genfiles__ = [Homebrew.bin('emu')]

  def install(self):
    Python().pip_install("emu", self.VERSION)


class JsonUtil(Task):
  """ json cli utils """
  __platforms__ = ['linux', 'osx']
  __deps__ = ['Node']
  __osx_deps__ = ['Homebrew']
  __genfiles__ = [Homebrew.bin('jsonlint')]
  __osx_genfiles__ = ['/usr/local/bin/jq']
  __linux_genfiles__ = ['/usr/bin/jq']
  __versions__ = {
      "jsonlint": "1.6.2",
  }

  def install(self):
    Homebrew().install_package("jq")
    Node().npm_install("jsonlint", self.__versions__["jsonlint"])

  def upgrade_osx(self):
    Homebrew().upgrade_package("jq")


class Scripts(Task):
  """ scripts and utils """
  __platforms__ = ['linux', 'osx']
  __genfiles__ = [
      '~/.local/bin/mkepisodal',
      '~/.local/bin/rm-dsstore',
      '~/.local/bin/mp3_transcode',
  ]

  def install(self):
    mkdir("~/.local/bin")
    symlink(usr_share("scripts/mkepisodal.py"), "~/.local/bin/mkepisodal")
    symlink(usr_share("scripts/rm-dsstore.sh"), "~/.local/bin/rm-dsstore")
    symlink(usr_share("scripts/mp3_transcode.sh"), "~/.local/bin/mp3_transcode")

  def uninstall(self):
    task_print("Removing scripts")
    Trash().trash(*self.__genfiles__)


class FlorenceScripts(Task):
  """Scripts just for florence."""
  __platforms__ = ['osx']
  __hosts__ = ['florence']
  __deps__ = ["Scripts"]
  __genfiles__ = [
      "~/.local/bin/orange_you_glad_you_backup",
  ]

  def install(self):
    symlink(
        usr_share("scripts/orange_you_glad_you_backup.sh"),
        "~/.local/bin/orange_you_glad_you_backup")

  def uninstall(self):
    task_print("Removing florence scripts")
    Trash().trash(*self.__genfiles__)


class LibExempi(Task):
  """ parse XMP metadata """
  __platforms__ = ['osx', 'linux']
  __deps__ = ['Homebrew']
  __genfiles__ = [Homebrew.lib('libexempi.a')]

  def install(self):
    Homebrew().install_package('exempi')


class LibMySQL(Task):
  __platforms__ = ['osx', 'linux']
  __deps__ = []

  def install(self):
    pass

  def install_linux(self):
    Apt().install_package('libmysqlclient-dev')


class Clang(Task):
  __platforms__ = ['linux', 'osx']
  __deps__ = ['Homebrew']
  __linux_genfiles__ = [Homebrew.bin('clang')]

  def install_osx(self):
    # LLVM comes free with macOS.
    pass

  def install_linux(self):
    Homebrew().install_package('llvm', '--with-libcxx')


class Ninja(Task):
  __platforms__ = ['linux', 'osx']
  __deps__ = ['Homebrew']
  __genfiles__ = [Homebrew.bin('ninja')]

  def install(self):
    Homebrew().install_package('ninja')

  def upgrade(self):
    Homebrew().upgrade_package('ninja')


class Rsync(Task):
  __platforms__ = ['linux', 'osx']
  __linux_genfiles__ = ['/usr/bin/rsync']

  def install_osx(self):
    # rsync comes free with macOS.
    pass

  def install_linux(self):
    Apt().install_package('rsync')


class InotifyMaxUserWatchers(Task):
  __platforms__ = ['linux']

  def install_linux(self):
    if not shell_ok('grep fs.inotify.max_user_watches /etc/sysctl.conf'):
      shell(
          "sudo sh -c 'echo fs.inotify.max_user_watches=1048576 >> /etc/sysctl.conf'"
      )
      shell('sudo sysctl -p')


class PhdBuildDeps(Task):
  """ phd repo dependencies"""
  __platforms__ = ['linux', 'osx']
  __deps__ = [
      'Bazel',
      'GitLfs',
      'GnuCoreutils',
      'LaTeX',
      'LibExempi',
      'LibMySQL',
      'Python',
      'Rsync',
  ]
  __linux_deps__ = [
      'InotifyMaxUserWatchers',
  ]
  __osx_deps__ = [
      # Needed by //labm8:hashcache.
      'GnuCoreutils',
  ]

  def install(self):
    pass


class Phd(Task):
  """PhD repo"""
  __platforms__ = ['linux', 'osx']
  __genfiles__ = ['~/phd/.env']
  __deps__ = ['PhdBuildDeps']

  def install(self):
    clone_git_repo(github_repo("ChrisCummins", "phd"), "~/phd")


class PhdDevDeps(Task):
  __platforms__ = ['linux', 'osx']
  __deps__ = [
      'Buildifier',
      'Homebrew',
      'Node',
      'Phd',
  ]
  __linux_deps__ = [
      'InotifyMaxUserWatchers',
  ]

  def install(self):
    shell('cd {phd} && {npm} install husky --save-dev'.format(
        npm=Node.NPM_BINARY, phd=os.path.expanduser('~/phd')))


class TransmissionHeadless(Task):
  """Headless bittorrent client."""
  __platforms__ = ['linux']
  __hosts__ = ['ryangosling']
  __linux_genfiles__ = [
      '/usr/share/transmission',
      '/usr/bin/transmission-cli',
      '/etc/transmission-daemon',
      '/usr/bin/transmission-daemon',
  ]

  def install(self):
    shell("sudo add-apt-repository -y ppa:transmissionbt/ppa")
    Apt().update()
    Apt().install_package("transmission-cli")
    Apt().install_package("transmission-common")
    Apt().install_package("transmission-daemon")


class TransmissionConfig(Task):
  """User config for transmission."""
  CFG = '/etc/transmission-daemon/settings.json'

  __platforms__ = ['linux']
  __hosts__ = ['ryangosling']
  __linux_genfiles__ = [CFG]
  __linux_deps__ = ['TransmissionHeadless']

  def install(self):
    if not os.path.islink(self.CFG):
      shell('sudo service transmission-daemon stop')
      symlink(
          '{private}/transmission/settings.json'.format(private=PRIVATE),
          self.CFG,
          sudo=True)
      shell('sudo service transmission-daemon start')


class DefaultApps(Task):
  """ set default applications for file extensions """
  # Use `duti -x epub` to list current file associations.
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
      "bst": "com.sublimetext.3",
      "c": "com.sublimetext.3",
      "cls": "com.sublimetext.3",
      "cpp": "com.sublimetext.3",
      "cxx": "com.sublimetext.3",
      "gz": "cx.c3.theunarchiver",
      "log": "com.sublimetext.3",
      "markdown": "com.sublimetext.3",
      "md": "com.sublimetext.3",
      "mkv": "org.videolan.vlc",
      "mov": "org.videolan.vlc",
      "mp4": "org.videolan.vlc",
      "mpg": "org.videolan.vlc",
      "nfo": "com.sublimetext.3",
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
      shell('duti -s {app} .{extension} all'.format(
          app=app, extension=extension))

  def upgrade(self):
    Homebrew().upgrade_package("duti")


class GoogleChrome(Task):
  """ internet explorer """
  __platforms__ = ['osx']
  __deps__ = ['Homebrew']

  def install_osx(self):
    Homebrew().install_cask('google-chrome')

  def upgrade_osx(self):
    Homebrew().upgrade_cask('google-chrome')


class Ripgrep(Task):
  """ a very, very fast grep """
  __platforms__ = ['osx', 'linux']
  __deps__ = ['Homebrew', 'Ruby']
  __genfiles__ = [Homebrew.bin("rg")]

  def install(self):
    if not shell_ok("gem list --local | grep rails"):
      task_print("gem install rails")
      shell("sudo gem install rails")
    Homebrew().install_package("ripgrep")


class Tower(Task):
  __platforms__ = ['osx']
  __osx_deps__ = ['Homebrew']
  __genfiles__ = ['/Applications/Tower.app']

  def install_osx(self):
    Homebrew().install_cask('tower')


class DnsTest(Task):
  """dns performance test"""
  __platforms__ = ['osx', 'linux']
  __osx_deps__ = ['Wget']
  __genfiles__ = ['~/.local/bin/dnstest']
  __versions__ = {
      'dnstest': '80341abdd2afc12cd18ce6404bb3c937b16ccfa7',
  }

  def install(self):
    url = ('https://raw.githubusercontent.com/cleanbrowsing/dnsperftest/'
           '{version}/dnstest.sh'.format(version=self.__versions__['dnstest']))
    if not os.path.isfile(os.path.expanduser('~/.local/bin/dnstest')):
      mkdir("~/.local/bin")
      shell("wget '{url}' -O ~/.local/bin/dnstest".format(url=url))
      shell('chmod +x ~/.local/bin/dnstest')


class PlatformIO(Task):
  """ Command line tools for building Arduino code. """
  # See https://platformio.org/
  __platforms__ = ['linux', 'osx']
  __deps__ = ['Python']
  __genfiles__ = ['/usr/local/bin/platformio']
  __versions__ = {'platformio': '3.6.3'}

  def install_linux(self):
    self.install()
    # Linuxbrew installs binaries in ~linuxbrew/.linuxbrew/bin, but the
    # bazel platformio rules are hardcoded to use only path /usr/local/bin.
    symlink('/usr/local/bin/platformio',
            '/home/linuxbrew/.linuxbrew/bin/platformio')

  def install(self):
    Python().pip_install("autoenv", self.__versions__['platformio'])

  def uninstall_linux(self):
    # Remove the symlink we created.
    os.unlink('/usr/local/bin/platformio')

class FinderGo(Task):
  __platforms__ = ['osx']
  __versions__ = {'FinderGo': '1.4.0'}

  def __init__(self):
    self.installed = False

  def install(self):
    url = ('https://github.com/onmyway133/FinderGo/releases/download/'
           '{version}/FinderGo.zip'.format(version=self.__versions__['FinderGo']))
    if not os.path.isfile('/Applications/FinderGo.app'):
      shell("wget '{url}' -O /Applications/FinderGo.zip".format(url=url))
      shell("cd /Applications && unzip FinderGo.zip && rm FinderGo.zip")
      self.installed = True

  def teardown(self):
    if self.installed:
      logging.info("")
      logging.info(
          "NOTE: manual step required to complete FinderGo installation:")
      logging.info(
          "    " + Colors.BOLD + Colors.RED +
          "Cmd+click and drag /Applications/FinderGo.app into Finder toolbar"
          + Colors.END)


class Bat(Task):
  __platforms__ = ['osx']

  def install(self):
    Homebrew().install_package("bat")


class Autojump(Task):
  __platforms__ = ['osx']

  def install(self):
    Homebrew().install_package("autojump")


class Fselect(Task):
  # https://github.com/jhspetersson/fselect
  __platforms__ = ['osx']

  def install(self):
    Homebrew().install_package("fselect")


class Glances(Task):
  # https://github.com/nicolargo/glances
  __platforms__ = ['osx', 'linux']

  def install(self):
    shell("curl -L https://bit.ly/glances | /bin/bash")


class Mycli(Task):
  # https://github.com/nicolargo/glances
  __platforms__ = ['osx', 'linux']

  def install(self):
    Python().pip_install('mycli', '1.19.0', sudo=True)


class DbBrowser(Task):
  # https://sqlitebrowser.org/
  __platforms__ = ['osx']

  def install(self):
    Homebrew().install_cask("db-browser-for-sqlite")
