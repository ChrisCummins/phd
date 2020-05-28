-- The type of an intermediate representation.

CREATE TYPE IR_TYPE AS ENUM('LLVM', 'XLA_HLO');

-- Compiler options used to generate an IR.

CREATE TABLE ir_opts(-- The numeric ID of the IR.
 ir_opts_id SERIAL PRIMARY KEY, -- A list of compiler options used to produce this IR, e.g. ['-O0'].
 -- The meaning of the options will depend on the compiler.
 copts VARCHAR(1024) UNIQUE --
);

-- A compiler intermediate representation (IR).
--
-- IRs are vertically partitioned across three tables: this table contains the
-- metadata describing the IR, ir_opts describes the compiler options, and
-- ir_text table contains the IR text.
 -- The text of an IR, kept in a separate table to maximize metadata query
-- performance.

CREATE TABLE ir_text (sha256 VARCHAR(64) PRIMARY KEY, -- The value of the IR.
 text_size INTEGER NOT NULL, -- The size of the IR text.
 line_count INTEGER NOT NULL, -- The number of lines in the IR text.
 text TEXT NOT NULL -- The full contents of the IR.
);


CREATE TABLE ir
  (ir_id BIGSERIAL PRIMARY KEY, -- A numeric ID for this IR.
 ir_type IR_TYPE NOT NULL, -- The type of the IR.
 ir_version VARCHAR(8) NOT NULL, -- The IR version.
 -- The compiler options used to generate the IR. If not known, or if no
 -- options are available, this may be NULL.
 ir_opts_id SERIAL REFERENCES ir_opts(ir_opts_id), -- The primary source file used to produce the IR. Additional secondary
 -- source files may be used, but are not recorded here.
 src_id BIGSERIAL REFERENCES src(src_id), -- A SHA256 checksum of the IR text, used for identifying duplicate IRs.
 sha256 VARCHAR(64) NOT NULL REFERENCES ir_text(sha256) ON DELETE CASCADE ON UPDATE CASCADE, -- The date that the IR was added to the table.
 date_added TIMESTAMP NOT NULL DEFAULT NOW(),
                                       UNIQUE(ir_type, ir_version, ir_opts_id, src_id) -- An IR is uniquely identified by its type, the compiler options, and the primary source file.
 );
