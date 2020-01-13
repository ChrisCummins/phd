import os

from experimental.util.dotfiles.implementation import host
from experimental.util.dotfiles.implementation import io
from experimental.util.dotfiles.implementation import task
from experimental.util.dotfiles.implementation.tasks import apt


class Homebrew(object):
  """ homebrew package manager """

  # Temporary files for caching list of installed packages and casks
  PKG_LIST = os.path.abspath(".brew-pkgs.txt")
  CASK_LIST = os.path.abspath(".brew-casks.txt")
  OUTDATED_PKG_LIST = os.path.abspath(".brew-pkgs-outdated.txt")
  OUTDATED_CASK_LIST = os.path.abspath(".brew-casks-outdated.txt")

  BREW_BINARY = {
    "osx": "/usr/local/bin/brew",
    "ubuntu": "/home/linuxbrew/.linuxbrew/bin/brew",
  }[host.GetPlatform()]

  __platforms__ = ["linux", "osx"]
  __deps__ = []
  __genfiles__ = [BREW_BINARY]
  __tmpfiles__ = [PKG_LIST, CASK_LIST, OUTDATED_PKG_LIST, OUTDATED_CASK_LIST]

  @classmethod
  def install(cls):
    if host.GetPlatform() == "osx":
      cls.install_osx()
    elif host.GetPlatform() == "ubuntu":
      cls.install_ubuntu()

  @staticmethod
  def install_osx():
    if not host.Which("brew"):
      io.Print("Installing Homebrew")
      url = "https://raw.githubusercontent.com/Homebrew/install/master/install"
      host.ShellCommand(
        'yes | /usr/bin/ruby -e "$(curl -fsSL {url})"'.format(**vars())
      )
      host.ShellCommand("brew doctor")

  @classmethod
  def install_ubuntu(cls):
    # Install build dependencies:
    apt.Apt().install_package("build-essential")
    apt.Apt().install_package("curl")
    apt.Apt().install_package("file")
    apt.Apt().install_package("git")
    apt.Apt().install_package("python-setuptools")

    if not os.path.exists("/home/linuxbrew/.linuxbrew/bin/brew"):
      url = (
        "https://raw.githubusercontent.com/"
        "Linuxbrew/install/master/install.sh"
      )
      host.ShellCommand('yes | sh -c "$(curl -fsSL {url})"'.format(url=url))
      host.ShellCommand("{brew} update".format(brew=cls.BREW_BINARY))

  def package_is_installed(self, package):
    """ return True if package is installed """
    if not os.path.isfile(self.PKG_LIST):
      host.ShellCommand(
        "{self.BREW_BINARY} list > {self.PKG_LIST}".format(**vars())
      )
    return host.CheckShellCommand(
      "grep '^{package}$' <{self.PKG_LIST}".format(**vars())
    )

  def install_package(self, package):
    """ install a package using homebrew, return True if installed """
    if not self.package_is_installed(package):
      io.Print("brew install " + package)
      host.ShellCommand("{self.BREW_BINARY} install {package}".format(**vars()))
      return True

  def package_is_outdated(self, package):
    """ returns True if package is outdated """
    if not self.package_is_installed(package):
      raise task.InvalidTaskError(
        "homebrew package '{package}' cannot be upgraded "
        "as it is not installed".format(**vars())
      )

    if not os.path.isfile(self.OUTDATED_PKG_LIST):
      host.ShellCommand(
        "{self.BREW_BINARY} outdated | awk '{{print $1}}' >{self.OUTDATED_PKG_LIST}".format(
          **vars()
        )
      )

    package_stump = package.split("/")[-1]
    return host.CheckShellCommand(
      "grep '^{package_stump}$' <{self.OUTDATED_PKG_LIST}".format(**vars())
    )

  def upgrade_package(self, package):
    """ upgrade package, return True if upgraded """
    if self.package_is_outdated(package):
      io.Print("brew upgrade {package}".format(**vars()))
      host.ShellCommand("{self.BREW_BINARY} upgrade {package}".format(**vars()))
      return True

  def cask_is_installed(self, cask):
    """ return True if cask is installed """
    if not os.path.isfile(self.CASK_LIST):
      host.ShellCommand(
        "{self.BREW_BINARY} cask list > {self.CASK_LIST}".format(**vars())
      )

    cask_stump = cask.split("/")[-1]
    return host.CheckShellCommand(
      "grep '^{cask_stump}$' <{self.CASK_LIST}".format(**vars())
    )

  def install_cask(self, cask):
    """ install a homebrew cask, return True if installed """
    if not self.cask_is_installed(cask):
      io.Print("brew cask install " + cask)
      host.ShellCommand(
        "{self.BREW_BINARY} cask install {cask}".format(**vars())
      )
      return True

  def cask_is_outdated(self, cask):
    """ returns True if cask is outdated """
    if not self.cask_is_installed(cask):
      raise task.InvalidTaskError(
        "homebrew cask '{package}' cannot be upgraded as it is not installed".format(
          **vars()
        )
      )

    if not os.path.isfile(self.OUTDATED_CASK_LIST):
      host.ShellCommand(
        "{self.BREW_BINARY} cask outdated ".format(**vars())
        + "| awk '{{print $1}}' >{self.OUTDATED_CASK_LIST}".format(**vars())
      )

    cask_stump = cask.split("/")[-1]
    return host.CheckShellCommand(
      "grep '^{cask_stump}$' <{self.OUTDATED_CASK_LIST}".format(**vars())
    )

  def upgrade_cask(self, cask):
    """ upgrade a homebrew cask. does nothing if cask not installed """
    if self.cask_is_outdated(cask):
      io.Print("brew cask upgrade {cask}".format(**vars()))
      host.ShellCommand(
        "{self.BREW_BINARY} cask upgrade {cask}".format(**vars())
      )
      return True

  def uninstall_cask(self, cask):
    """ remove a homebrew cask, return True if uninstalled """
    if self.cask_is_installed(cask):
      io.Print("brew cask remove " + cask)
      host.ShellCommand("{self.BREW_BINARY} cask remove " + cask)
      return True

  @staticmethod
  def _home():
    if host.GetPlatform() == "osx":
      return "/usr/local"
    else:
      return "/home/linuxbrew/.linuxbrew"

  @classmethod
  def bin(cls, name):
    home = cls._home()
    return "{home}/bin/{name}".format(home=home, name=name)

  @classmethod
  def lib(cls, name):
    home = cls._home()
    return "{home}/lib/{name}".format(home=home, name=name)


class brew_package(task.Task):
  def __init__(
    self, name=None, genfiles=None, deps=None, package=None, force_link=None
  ):
    super(brew_package, self).__init__(name, genfiles, deps)
    self.package = task.AssertSet(package, "package")
    self.force_link = force_link

  def __call__(self, ctx):
    package = self.package
    brew = Homebrew()
    brew.install()
    if brew.install_package(package):
      host.ShellCommand(
        "{brew.BREW_BINARY} link {package} --force".format(
          brew=brew, package=package
        )
      )


def brew_bin(name):
  return Homebrew.bin(name)
