--
-- Fetched
--

CREATE TABLE IF NOT EXISTS Repositories (
       url                 TEXT NOT NULL,       -- GitHub URL
       owner               TEXT,                -- Email
       name                TEXT NOT NULL,       -- GitHub Repository name
       fork                INT NOT NULL,        -- 1 if repo is fork, else 0
       stars               INT NOT NULL,        -- GitHub stargazer count
       contributors        INT NOT NULL,        -- Number of contributors
       forks               INT NOT NULL,        -- Number of forks
       created_at          DATETIME NOT NULL,   -- Timestamp created
       updated_at          DATETIME NOT NULL,   -- Timestamp last updated
       UNIQUE(url)
);

CREATE TABLE IF NOT EXISTS OpenCLFiles (
       url                 TEXT NOT NULL,       -- GitHub URL
       path                TEXT NOT NULL,       -- File path within repo
       repo_url            TEXT NOT NULL,       -- Repositories.url
       contents            TEXT NOT NULL,       -- File contents
       sha                 TEXT NOT NULL,       -- File checksum
       size                INT NOT NULL,        -- File size
       UNIQUE(url)
);

--
-- Preprocessed
--

CREATE TABLE IF NOT EXISTS Preprocessed (
       sha                 TEXT NOT NULL,       -- OpenCLFile.sha
       contents            TEXT NOT NULL,       -- Preprocessed
       UNIQUE(sha)
);

CREATE TABLE IF NOT EXISTS PreprocessedErrors (
       sha                 TEXT NOT NULL,       -- OpenCLFile.sha
       message             TEXT NOT NULL,       -- Compilation error
       UNIQUE(sha)
);

CREATE TABLE IF NOT EXISTS Bytecodes (
       sha                 TEXT NOT NULL,       -- OpenCLFile.sha
       contents            TEXT NOT NULL,       -- Bytecode
       UNIQUE(sha)
);

CREATE TABLE IF NOT EXISTS BytecodeErrors (
       sha                 TEXT NOT NULL,       -- OpenCLFile.sha
       message             TEXT NOT NULL,       -- Compilation error
       UNIQUE(sha)
);
