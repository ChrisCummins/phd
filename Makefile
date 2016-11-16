# This file is part of cecl.
#
# cecl is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cecl is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cecl.  If not, see <http://www.gnu.org/licenses/>.
#
SYSTEM := $(shell uname -s)

PREFIX ?= /usr/local

ifneq ($(SYSTEM),Darwin)
OPENCL ?= -I/usr/local/cuda/include
endif

AR ?= ar
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra

all: libcecl.a

cecl.o: cecl.c cecl.h
	$(CC) $(CFLAGS) $(OPENCL) $< -c -o $@

libcecl.a: cecl.o
	$(AR) rcs $@ $?

.PHONY: clean install uninstall

clean:
	rm -f cecl.o
	rm -f libcecl.a

install: all mkcecl runcecl
	cp cecl.h $(PREFIX)/include/cecl.h
	cp libcecl.a $(PREFIX)/lib/libcecl.a
	cp mkcecl $(PREFIX)/bin/mkcecl
	cp runcecl $(PREFIX)/bin/runcecl
	chmod +x $(PREFIX)/bin/mkcecl $(PREFIX)/bin/runcecl

uninstall:
	rm -f $(PREFIX)/include/cecl.h
	rm -f $(PREFIX)/lib/libcecl.a
	rm -f $(PREFIX)/bin/mkcecl
	rm -f $(PREFIX)/bin/runcecl
