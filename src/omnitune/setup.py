from setuptools import setup

# TODO: Set license.
# TODO: Set description.
setup(name='omnitune',
      version='0.0.1',
      description='',
      url='https://github.com/ChrisCummins/msc-thesis',
      author='Chris Cummins',
      author_email='chrisc.101@gmail.com',
      license='',
      packages=[
          'omnitune',
          'omnitune.skelcl'
      ],
      scripts=['bin/omnitune-proxy'],
      test_suite='nose.collector',
      tests_require=[
          'nose'
      ],
      install_requires=[
          'labm8',
          'pyopencl',
          'python-weka-wrapper'
      ],
      data_files=[
          ('/etc/dbus-1/system.d', ['etc/org.omnitune.conf'])
      ],
      zip_safe=False)
