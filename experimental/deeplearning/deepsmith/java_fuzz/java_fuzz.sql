# MySQL snippets for working with java_fuzz contentfiles database.
 # How many active repos?

SELECT count(*)
FROM repositories
WHERE active = 1;

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

