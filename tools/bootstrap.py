# TODO(cec): Implement!
  # logging.info('Installing Python packages')
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
