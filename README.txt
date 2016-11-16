                 cecl - OpenCL Application profiling
                 ===================================

What is cecl?

  A small library which provides replacement calls for a subset of the
  OpenCL API. It allows simple performance benchmarking of OpenCL programs.

Installation:

  $ make
  $ sudo make install

Usage:

  Rewrites the OpenCL calls in myapp.cpp to use cecl:

    mkcecl myapp.cpp

  Compile and link myapp using libcecl:

    g++ myapp.cpp -o myapp -lcecl -lOpenCL

  Execute compiled application and print profiling information:

    runcecl ./myapp

Requirements:

  Linux or OS X.
  OpenCL 1.2.
  A working C compiler.

License:

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see
  <http://www.gnu.org/licenses/>.
