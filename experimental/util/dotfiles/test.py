brew_package(
  name="vim_app", outs=[brew_path("bin/vim")], package="vim",
)

symlink(
  name="vimrc", src="~/Dropbox/Shared/vim/vimrc", dst="~/.vimrc",
)

task_group(
  name="vim", deps=["brew_vim"],
)

brew_package(
  name="python2_app",
  force_link=True,
  outs_linux=[brew_path("bin/python2")],
  outs_osx=["/usr/local/opt/python@2/bin/python2"],
  package="python@2",
)

task_group(
  name="python", deps=["pypirc", "python2", "python3",],
)

task_group(
  name="python2", deps=["brew_python2"],
)

brew_package(
  name="python3",
  outs=[brew_path("bin/python3")],
  force_link=True,
  package="python",
)

symlink(
  name="pypirc", src="~/Dropbox/Shared/python/.pypirc", dst="~/.pypirc",
)

brew_package(
  name="unzip",
  outs=[brew_path("bin/unzip")],
  # When "package" is not set, default to "name".
)

brew_cask(
  name="omnifocus_app", outs=["/Applications/OmniFocus.app"], platforms=["osx"],
)

task_group(
  name="omnifocus", deps=["omnifocus_app", "omni", "of2"],
)

symlink(
  name="omni",
  src="~/Dropbox/Shared/omnifocus/omni",
  dst="/usr/local/bin/omni",
  use_sudo=True,
)


def NeedToInstallOf2():
  return (
    os.path.exists("/usr/local/opt/ofexport/bin/of2")
    and shell("/usr/local/opt/ofexport/bin/of2 -h").split("\n")[2]
    != "Version: 1.0.20"
  )


genrule(
  name="of2",
  outs=["/usr/local/opt/ofexport/bin/of2"],
  # Commands is a list of shell commands or python functions.
  cmds=[
    "rm -rf /usr/local/opt/ofexport",
    "wget https://github.com/psidnell/ofexport2/archive/ofexport-v2-1.0.20 -O /tmp/ofexport.zip",
    "unzip -o /tmp/ofexport.zip",
    "rm -f /tmp/ofexport.zip",
    "mv ofexport2-ofexport-v2-1.0.20 /usr/local/opt/ofexport",
  ],
  # Reqs is a list of shell commands or python functions.
  reqs=[NeedToInstallOf2],
)

brew_package(
  name="bazel_brew_app",
  platforms=["osx"],
  package="bazel",
  outs=[brew_path("bin/bazel")],
)

genrule(
  name="bazel_apt_repo",
  cmds=[
    (
      "echo 'deb [arch=amd64] http://storage.googleapis.com/bazel-apt' "
      "stable jdk1.8 | sudo tee /etc/apt/sources.list.d/bazel.list"
    ),
    "curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -",
  ],
)

apt_package(
  name="bazel_apt_app",
  platforms=["linux"],
  deps=["bazel_apt_repo", "java"],
  package="bazel",
  outs=[apt_path("bin/bazel")],
)

task_group(name="bazel", deps=["bazel_brew_app", "bazel_apt_app"])

brew_package(
  name="clang",
  platforms=["linux"],
  pacakge="llvm",
  outs=[brew_path("bin/clang")],
)

genrule(
  name="phd_bootstrap",
  outs=["~/phd/.env"],
  deps=["phd_repo", "bazel", "clang"],
)
