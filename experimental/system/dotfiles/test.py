brew_package(
  name="brew_vim",
  package="vim",
  genfiles=[brew_bin("vim")]
)

task_group(
  name="vim",
  deps=["brew_vim"],
)

brew_package(
  name="python2",
  package="python@2",
  force_link=True,
  genfiles=[brew_bin("python2")]
)

brew_package(
  name="python3",
  package="python",
  force_link=True,
  genfiles=[brew_bin("python3")]
)

symlink(
  name="pypirc",
  src='~/Dropbox/Shared/python/.pypirc',
  dst="~/.pypirc"
)

brew_package(
  name="unzip",
  package="unzip",
  genfiles=[brew_bin('unzip')]
)
