-- A source language.

CREATE TYPE SRC_LANGUAGE AS ENUM ('UNKNOWN', 'C', 'CXX', 'OPENCL', 'SWIFT', 'HASKELL', 'FORTRAN');

-- The metadata describing source files.
--
-- A source file is an input to a compiler. Source files are found in git
-- repositories. Many files can be extracted from a single repository. There is
-- no requirement that *all* source files from a repository exist in this
-- table.
--
-- Source files are vertically partitioned across two tables: this table
-- contains the metadata describing the source file, and the src_text table
-- contains the text of sources, when available.

CREATE TABLE src
  (-- A numeric ID for a source file.
 src_id BIGSERIAL PRIMARY KEY, -- The git repository that this source was found in.
 repo_id BIGSERIAL REFERENCES repo(repo_id) ON DELETE CASCADE ON UPDATE CASCADE, -- The relative path of the file within the git repository.
 relpath VARCHAR(255) NOT NULL, -- The source language of this file.
 src_language SRC_LANGUAGE NOT NULL, -- A SHA256 checksum of the source text, used for identifying duplicate
 -- source files. This column must be set when src_text exists.
 sha256 VARCHAR(64), -- The date that this source file was added to the table.
 date_added TIMESTAMP NOT NULL DEFAULT NOW(), -- A source file is unique within a git repository.
 UNIQUE (repo_id,
         relpath));

-- The text of a source file, kept in a separate table to maximize metadata
-- query performance.

CREATE TABLE src_text
  (-- The numeric ID of the source file.
 src_id BIGSERIAL PRIMARY KEY REFERENCES src(src_id) ON DELETE CASCADE ON UPDATE CASCADE, -- The full, unmodified contents of the source file.
 text TEXT NOT NULL);
