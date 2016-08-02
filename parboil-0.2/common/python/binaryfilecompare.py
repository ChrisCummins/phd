# (c) Copyright 2007 The Board of Trustees of the University of Illinois.

import struct

def uint16(f):
	"""Read a 16-bit unsigned integer from f."""
        chars = f.read(2)
	if len(chars) != 2: raise ValueError, "Unexpected end of file"
	[c, c2] = chars
	return ord(c) + (ord(c2) << 8)

def uint32(f):
	"""Read a 32-bit unsigned integer from f."""
	chars = f.read(4)
	if len(chars) != 4: raise ValueError, "Unexpected end of file"
	[c, c2, c3, c4] = chars
	return ord(c) + (ord(c2) << 8) + (ord(c3) << 16) + (ord(c4) << 24)

def float(f):
	"""Read a floating-point number from f."""
	s = f.read(4)
	if len(s) != 4: raise ValueError, "Unexpected end of file"
	(n,) = struct.unpack("<f", s)
	return n

def many(reader, count):
	"""Create a reader function that reads a fixed-size sequence of
	values from f."""
	return lambda f: [reader(f) for n in range(count)]

def many_uint16(count):
        """Create a reader function that reads a fixed-size sequence of
        16-bit unsigned integers from f."""
        def read(f):
                s = f.read(2*count)
		if len(s) != 2*count:
			raise ValueError, "Unexpected end of file"
                ns = struct.unpack("<%dH" % count, s)
                return ns
        return read

def many_float(count):
        """Create a reader function that reads a fixed-size sequence of
        floats from f."""
        def read(f):
                s = f.read(4*count)
		if len(s) != 4*count:
			raise ValueError, "Unexpected end of file"
                floats = struct.unpack("<%df" % count, s)
                return floats
        return read

def eof(f):
	"""Read the end of file 'f'.  A ValueError is raised if anything
	other than EOF is read from the file."""
	if f.readline() == "": return None
	else: raise ValueError, "Expecting end of file"


