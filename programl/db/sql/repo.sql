-- A git repository.

CREATE TABLE repo (-- A numeric ID for this repository.
 repo_id BIGSERIAL PRIMARY KEY,
                           url VARCHAR(2000) NOT NULL, -- The "git clone" URL of this repository, or URL of an archive.
 sha1 CHAR(40), -- The checksum of the commit (if a git repo), or of the file (if an archive).
 created_date TIMESTAMP NOT NULL, -- The date that this repository was cloned. This column can be used to
 -- periodically check for repositories that should be re-scanned for updates.
 date_added TIMESTAMP NOT NULL DEFAULT NOW(), -- A git repo is uniquely identified by its state at the time that it was
 -- cloned.
 UNIQUE (url,
         sha1)--
);
