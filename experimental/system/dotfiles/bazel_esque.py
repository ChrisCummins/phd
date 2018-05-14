# The idea is to sketch out a Bazel-esque approach to specifying dotfiles,
# using specific functions instead of ad-hoc classes.

task_group(
    name = "vim",
    deps = [":vim-brew", ":vimrc"],
)

brew_package(
    name = "vim-brew",
    package = "vim",
)

symlink(
    name = "vimrc",
    src = "~/Dropbox/Shared/vim/vimrc",
    dst = "~/.vimrc",
    deps = [":dropbox"]
)

symlink(
    name = "pypirc",
    src = "~/Dropbox/Shared/python/.pypirc",
    dst = "~/.pypirc",
)

github_repo(
    name = "me.csv",
    user = "ChrisCummins",
    repo = "me.csv",
    dst = "~/src/me.csv",
    head = "123adf78368000",
)

task_group(
    name = "python",
    deps = [":python2.7", ":python3.6", ":piprc"],
)

brew_package(
    name = "python2.7",
    package = "vim@2",
)

symlink(
    name = "piprc",
    src = "~/Dropbox/Shared/...",
    dst = "~/.piprc",
    deps = [":dropbox"]
)

shell(
    name = "some_example",
    src = [
        "# list of totally ad-hoc shell commands",
    ]
)
