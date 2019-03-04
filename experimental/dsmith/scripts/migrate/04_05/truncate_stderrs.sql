# Query to truncat stderrs to 64k chars, and set linecount and charcount.

UPDATE stderrs
SET stderr=substring(stderr, 1, 64000),
    charcount=char_length(stderr),
    linecount=char_length(stderr) - char_length(replace(stderr, '\n', '')) + 1,
    truncated=CASE
                  WHEN char_length(stderr) > 64000 THEN 1
                  ELSE 0
              END;

