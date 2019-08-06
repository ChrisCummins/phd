# Rules which embed files as strings in source files.


def _python_string_genrule_impl(ctx):
  in_file = ctx.file.src
  out_file = ctx.actions.declare_file(ctx.file.out.path)
  ctx.actions.run_shell(
      command=
      "echo 'STRING = \"\"\"' > {out_file}; cat '{in_file}' >> '{out_file}'; echo '\"\"\"' >> {out_file}"
      .format(in_file=in_file.path, out_file=out_file.path),
      inputs=[in_file],
      outputs=[out_file],
      progress_message="Creating %s" % out_file.short_path,
  )
  return [
      DefaultInfo(files=depset([out_file]),
                  runfiles=ctx.runfiles(files=[out_file])),
      PyInfo(transitive_sources=depset([out_file]))
  ]


python_string_genrule = rule(implementation=_python_string_genrule_impl,
                             attrs={
                                 "out":
                                 attr.label(mandatory=True,
                                            allow_single_file=True),
                                 "src":
                                 attr.label(mandatory=True,
                                            allow_single_file=True),
                             })
