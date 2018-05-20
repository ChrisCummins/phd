-- Dataset with GitHub metadata
CREATE TABLE IF NOT EXISTS Meta (
        key                        TEXT NOT NULL,
        value                      TEXT,
        UNIQUE(key)
);

--
-- Fetched
--

CREATE TABLE IF NOT EXISTS ContentFiles (
        id                  TEXT NOT NULL,       -- Content ID
        contents            TEXT NOT NULL,       -- Contents
        UNIQUE(id)
);


-- GitHub metadata
CREATE TABLE IF NOT EXISTS ContentMeta (
        id                  TEXT NOT NULL,       -- ContentFiles.ID
        path                TEXT NOT NULL,       -- File path within repo
        repo_url            TEXT NOT NULL,       -- Repositories.url
        sha                 TEXT NOT NULL,       -- Git checksum
        size                INT NOT NULL,        -- File size
        UNIQUE(id)
);

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

--
-- Preprocessed
--

CREATE TABLE IF NOT EXISTS PreprocessedFiles (
        id                 TEXT NOT NULL,       -- ContentFiles.id
        status             INTEGER NOT NULL,    -- 0 if good, 1 if bad, 2 if ugly
        contents           TEXT NOT NULL,       -- preprocess output
        UNIQUE(id)
);
