from setuptools import setup

APP = ['Timer.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    # 'iconfile': 'icon.icns',
    'plist': {
        'LSUIElement': True,
    },
    'packages': ['rumps'],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
