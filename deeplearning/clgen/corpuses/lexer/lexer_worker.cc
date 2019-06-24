// Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
//
// clgen is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// clgen is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with clgen.  If not, see <https://www.gnu.org/licenses/>.
#include "deeplearning/clgen/corpuses/lexer/lexer.h"
#include "deeplearning/clgen/proto/internal.pb.h"

#include "phd/pbutil.h"

PBUTIL_INPLACE_PROCESS_MAIN(clgen::ProcessLexerBatchJobOrDie,
                            clgen::LexerBatchJob);
