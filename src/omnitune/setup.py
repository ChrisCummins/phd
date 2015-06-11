from setuptools import setup

base_deps = [
    'labm8',
    'numpy',
    'pyopencl'
]

bells_and_whistles = True
extra_deps = [
    'matplotlib',
    'python-weka-wrapper',
    'seaborn'
]

if bells_and_whistles:
    deps = base_deps + extra_deps
else:
    deps = base_deps

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
      install_requires=deps,
      data_files=[
          ('/etc/dbus-1/system.d', ['etc/org.omnitune.conf'])
      ],
      zip_safe=False)
