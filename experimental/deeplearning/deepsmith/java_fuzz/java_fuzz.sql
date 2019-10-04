# MySQL snippets for working with java_fuzz contentfiles database.
 # What is the oldest scraped file?

SELECT *
FROM github_java.contentfiles
ORDER BY id
LIMIT 1;

# What is the most recently scraped file?

SELECT *
FROM github_java.contentfiles
ORDER BY id DESC
LIMIT 1;

# How many files have been scraped?

SELECT count(*)
FROM github_java.contentfiles;

# Select a handful of scraped files for inspection.

SELECT *
FROM github_java.contentfiles;

# How many repos did we scrape?

SELECT count(*)
FROM github_java.repositories;

# How many users' data did we scrape?

SELECT count(*)
FROM
  (SELECT distinct(OWNER)
   FROM github_java.repositories) t;

# What is the total line and character counts of the scraped Java?

SELECT sum(linecount),
       sum(charcount)
FROM github_java.contentfiles;

# How many `.java` files did we scrape?
# NOTE: This is an expensive query (full table scan).

SELECT count(*)
FROM github_java.contentfiles;

# Reset repositories masks.

UPDATE github_java.repositories
SET active = 1,
    exported = 0;

# After pruning, how many GitHub repos are active?

SELECT count(*)
FROM github_java.repositories
WHERE active = 1;

# How many `.java` files is that?

SELECT count(*)
FROM github_java.contentfiles
LEFT JOIN github_java.repositories ON github_java.contentfiles.clone_from_url=github_java.repositories.clone_from_url
WHERE github_java.repositories.active = 1;

# How many methods have been pre-processed?

SELECT count(*)
FROM `github_java_methods_2019.06.25`.repositories
WHERE exported=1;

# Reset pre-processed.

DELETE
FROM `github_java_methods_pp_2019.06.25`.preprocessed_contentfiles;


UPDATE `github_java_methods_2019.06.25`.repositories
SET exported=0;

# Top 10 most popular repos scraped.

SELECT name AS repo,
       OWNER AS USER,
                num_stars AS stars,
                num_forks AS forks,
                count(*) AS file_count
FROM `github_java_methods_2019.06.25`.contentfiles AS cf
LEFT JOIN `github_java_methods_2019.06.25`.repositories AS repo ON cf.clone_from_url=repo.clone_from_url
GROUP BY cf.clone_from_url
ORDER BY file_count DESC
LIMIT 10;

# Table of pre-processing results.

SELECT "Pass",
       count(*) AS num_methods,
       (count(*) /
          (SELECT count(*)
           FROM `github_java_methods_pp_2019.07.16`.preprocessed_contentfiles)) * 100 AS "% of total"
FROM `github_java_methods_pp_2019.07.16`.preprocessed_contentfiles
WHERE preprocessing_succeeded=1
UNION
SELECT text, count(*) AS num_methods,
             (count(*) /
                (SELECT count(*)
                 FROM `github_java_methods_pp_2019.07.16`.preprocessed_contentfiles)) * 100 AS "% of total"
FROM `github_java_methods_pp_2019.07.16`.preprocessed_contentfiles
WHERE preprocessing_succeeded = 0
GROUP BY text
ORDER BY num_methods DESC;

# Select some preprocessed files for inspection.

SELECT *
FROM `github_java_methods_pp_2019.06.28`.preprocessed_contentfiles
WHERE preprocessing_succeeded = 1;

# Reset encoded.

DELETE
FROM `github_java_methods_enc_2019.06.25`.encoded_contentfiles;


# Copy only the preprocessed methods which don't include "String" keyword.

INSERT INTO `github_java_methods_no_strings_pp_2019.07.16`.preprocessed_contentfiles
SELECT * FROM `github_java_methods_pp_2019.07.16`.preprocessed_contentfiles
WHERE text NOT LIKE "%String%";

# How many preprocessed methods include a string literal?

SELECT COUNT(*) FROM `github_java_methods_pp_2019.07.16`.preprocessed_contentfiles
WHERE preprocessing_succeeded = 1 AND text LIKE '%"%';

# How many of these non-string methods include a string literal?

SELECT COUNT(*) FROM `github_java_methods_no_strings_pp_2019.07.16`.preprocessed_contentfiles
WHERE preprocessing_succeeded = 1 AND text LIKE '%"%';