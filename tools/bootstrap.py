# TODO(cec): Implement!
# dotfiles_run_path = os.path.join(PHD_ROOT, 'system', 'dotfiles', 'run')
# assert os.path.isfile(dotfiles_run_path)
# proc = subprocess.Popen([dotfiles_run_path, 'PhdBuildDeps'])
# proc.communicate()
#
# pip_packages = RunAsHomebrewUser(
#     [python_path, '-m', 'pip', 'freeze'], subprocess.check_output).decode(
#     'utf-8')
# if args.with_cuda and 'tensorflow==' in pip_packages:
#   app.Info('Detected CUDA, so uninstalling CPU TensorFlow')
#   RunAsHomebrewUser(
#       [python_path, '-m', 'pip', 'uninstall', '-y', '-q', 'tensorflow'])
# elif not args.with_cuda and 'tensorflow-gpu==' in pip_packages:
#   app.Info('CUDA not detected, so uninstalling GPU TensorFlow')
#   RunAsHomebrewUser(
#       [python_path, '-m', 'pip', 'uninstall', '-y', '-q', 'tensorflow-gpu'])
#
# app.Info('Installing Python packages')
# # If the current user is root, then the linuxbrew user will not have
# # permissions to read the requirements.txt file. Therefore we create a
# # temproary copy of the requirements.txt and install using that.
# with tempfile.NamedTemporaryFile(
#     prefix='phd_python_requirements_', suffix='.txt') as f:
#   f.write(requirements_txt.encode('utf-8'))
#   f.flush()
#   # Change the permissions of the file to octal 0666.
#   os.chmod(f.name, 436)
#   RunAsHomebrewUser(
#       [python_path, '-m', 'pip', 'install', '--no-warn-script-location', '-q',
#        '-r', f.name])
