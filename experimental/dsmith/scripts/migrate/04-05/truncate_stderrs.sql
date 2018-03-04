# Query to truncat stderrs to 64k chars, and set linecount and charcount.
UPDATE stderrs
SET
    stderr=SUBSTRING(stderr, 1, 64000),
    charcount=CHAR_LENGTH(stderr),
    linecount=CHAR_LENGTH(stderr) - CHAR_LENGTH(REPLACE(stderr, '\n', '')) + 1,
    truncated=CASE WHEN CHAR_LENGTH(stderr) > 64000 THEN 1 ELSE 0 END;
