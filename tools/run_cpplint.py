#!/usr/bin/env bash
set - eu

tools_dir = "$( cd "$(dirname "${BASH_SOURCE[0]}")
" && pwd )"

cpplint_args = "--root=include --filter=-build/c++11build/header_guard,-build/include_order,-legal,-readability/streams,-readability/todo,-runtime/reference"

find. - name
'*.cc' | xargs
"$tools_dir/cpplint.py" $cpplint_args
2 > & 1 | grep - v
'^Done processing '
