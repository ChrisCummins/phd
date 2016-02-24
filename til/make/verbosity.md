# Controlling Makefile verbosity

To disable echoing of executed commands, prefix with
`@`. Alternatively, to enable printing echoing of commands when an
optional `V=`argument is passed, add the following two variables to
the top of the Makefile:

```
QUIET_ = @
QUIET = $(QUIET_$(V))
```

Then, prefix commands that you want to execute quietly by default with
`$(QUIET)`:

```
%.o: %.cpp
    @echo '  CC  $@'
    $(QUIET)clang++ $< -c -o $@
```

This will execute quietly except if invoked using the V argument:
`make V=1`.
