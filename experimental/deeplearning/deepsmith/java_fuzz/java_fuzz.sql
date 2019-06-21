# MySQL snippets for working with java_fuzz contentfiles database.
 # How many active repos?
 # Reset preprocessor.

DROP DATABASE `github_java_methods_pp`;


UPDATE github_java_methods.repositories
SET exported = 0;

# Count repos.

SELECT count(*)
FROM repositories
WHERE active = 0;


SELECT count(*)
FROM repositories
WHERE exported = 1;


SELECT count(*)
FROM github_java_methods.contentfiles
WHERE active = 1;


SELECT count(*)
FROM repositories
WHERE exported = 0;


UPDATE repositories
SET exported = 0;


SELECT *
FROM github_java.contentfiles
WHERE clone_from_url = 'https://github.com/kerstin/fhtw-hamster.git';


SELECT count(*)
FROM github_java_methods.contentfiles
WHERE active = 1;


SELECT *
FROM github_java.contentfiles
WHERE text LIKE '%-----BEGIN RSA PRIVATE KEY-----%'
LIMIT 1;


ALTER TABLE github_java_methods.contentfiles ADD active bool NOT NULL DEFAULT 1 AFTER artifact_index;


UPDATE repositories
SET active = 0;

# Top 10 most popular repos scraped.

SELECT name AS repo,
       OWNER AS USER,
                num_stars AS stars,
                num_forks AS forks,
                count(*) AS file_count
FROM contentfiles
LEFT JOIN repositories ON contentfiles.clone_from_url=repositories.clone_from_url
GROUP BY contentfiles.clone_from_url
ORDER BY file_count DESC
LIMIT 10;

# How many contentfiles if we require minimum 10 stars?

SELECT sum(file_count)
FROM
  (SELECT num_stars AS stars,
          count(*) AS file_count
   FROM contentfiles
   LEFT JOIN repositories ON contentfiles.clone_from_url=repositories.clone_from_url
   WHERE num_stars >= 10
   GROUP BY contentfiles.clone_from_url) t;

