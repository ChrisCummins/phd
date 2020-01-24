# libcecl - OpenCL Application profiling

Libcecl is a small library which provides synchronous, verbose wrappers for a
subset of the OpenCL API. It allows simple performance benchmarking of OpenCL
programs.

## Usage

Rewrites the OpenCL calls in a C/C++ source to use libcecl:

```py
src = libcecl_rewriter.RewriteOpenClSource(src)
```

Compile and link the rewritten source:

```py
cflags, ldflags = libcecl_compile.LibCeclCompileAndLinkFlags()
clang.Exec([src_file, '-o', str(binary)path)] + cflags + ldflags)
```

Execute compiled application and print profiling information:

```py
libcecl_runtime.RunLibceclExecutable(
    [binary_path], cldrive_env.OclgrindOpenCLEnvironment())
```

## License

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.
