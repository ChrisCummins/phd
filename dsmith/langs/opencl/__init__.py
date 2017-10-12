import dsmith
from dsmith.langs import Language


class Generator(object):
    pass


class CLSmith(Generator):
    __name__ = "clsmith"


class DSmith(Generator):
    __name__ = "dsmith"


class OpenCL(Language):
    __name__ = "opencl"

    __generators__ = {
        None: DSmith,
        "dsmith": DSmith,
        "clsmith": CLSmith,
    }

    def mkgenerator(self, string: str) -> Generator:
        generator = self.__generators__.get(string)
        if not generator:
            raise ValueError("Unknown generator")
        return generator()
