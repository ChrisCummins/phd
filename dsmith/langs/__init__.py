class Language(object):
    pass

# Deferred importing of languages, since the modules may need to import this
# file.
from dsmith.langs.opencl import OpenCL

LANGUAGES = {
    "opencl": OpenCL,
}


def mklang(string: str) -> Language:
    lang = LANGUAGES.get(string)
    if not lang:
        raise ValueError("Unknown language")
    return lang()
