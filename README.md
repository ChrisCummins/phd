# lmk - let me know

Email notifications from the command line.

**Step 1** Wrap your long running job in `lmk`:

```sh
$ lmk 'bash ./experiments.sh'
...  # command runs and outputs normally
[lmk] chrisc.101@gmail.com notified
```

**Step 2** ☕

**Step 3** Receive an email when it's done:

![](demo.png)

## Installation
```sh
$ pip install lmk
```

## License

Made with ♥ by [Chris Cummins](http://chriscummins.cc). Released under [MIT License](https://tldrlegal.com/license/mit-license).
