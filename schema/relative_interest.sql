DROP TABLE IF EXISTS major_subreddits;
CREATE TABLE major_subreddits AS (
  SELECT subreddit,
    COUNT(*) AS user_count
  FROM (
      SELECT DISTINCT subreddit,
        author
      FROM unique_comments
    ) AS a
  GROUP BY subreddit
  HAVING COUNT(*) >= 500
  ORDER BY user_count DESC
  LIMIT 50
);
DROP TABLE IF EXISTS relative_interest_by_comments;
CREATE TABLE relative_interest_by_comments AS (
  WITH x AS (
    SELECT mbti,
      subreddit,
      COUNT(*) AS comments_per_subreddit_by_mbti
    FROM unique_comments
    WHERE subreddit IN (
        SELECT subreddit
        FROM major_subreddits
      )
      AND NOT is_mbti_related
    GROUP BY mbti,
      subreddit
  ),
  x_tot AS (
    SELECT mbti,
      SUM(comments_per_subreddit_by_mbti) AS total_comments_by_mbti
    FROM x
    GROUP BY mbti
  ),
  x_f AS (
    SELECT x.mbti,
      x.subreddit,
      total_comments_by_mbti,
      comments_per_subreddit_by_mbti,
      ROUND(
        comments_per_subreddit_by_mbti * 1.0 / total_comments_by_mbti * 1.0,
        4
      ) AS share
    FROM x
      INNER JOIN x_tot ON x.mbti = x_tot.mbti
  )
  SELECT *
  FROM x_f
);
DROP TABLE IF EXISTS relative_interest_distinct_users;
CREATE TABLE relative_interest_distinct_users AS (
  WITH distinct_users AS (
    SELECT DISTINCT author,
      mbti,
      subreddit
    FROM unique_comments
    WHERE subreddit IN (
        SELECT subreddit
        FROM major_subreddits
      )
      AND NOT is_mbti_related
  ),
  x AS (
    SELECT mbti,
      subreddit,
      COUNT(*) AS users_per_subreddit_by_mbti
    FROM distinct_users
    GROUP BY mbti,
      subreddit
  ),
  x_tot AS (
    SELECT mbti,
      SUM(users_per_subreddit_by_mbti) AS total_users_by_mbti
    FROM x
    GROUP BY mbti
  ),
  x_f AS (
    SELECT x.mbti,
      x.subreddit,
      total_users_by_mbti,
      users_per_subreddit_by_mbti,
      ROUND(
        users_per_subreddit_by_mbti * 1.0 / total_users_by_mbti * 1.0,
        4
      ) AS share
    FROM x
      INNER JOIN x_tot ON x.mbti = x_tot.mbti
  )
  SELECT *
  FROM x_f
);
DROP TABLE IF EXISTS relative_interest_grouped;
CREATE TABLE relative_interest_grouped AS (
  WITH distinct_users AS (
    SELECT DISTINCT author,
      mbti,
      g.group_name AS subreddit_group
    FROM unique_comments AS c
      INNER JOIN subreddit_groups AS g ON c.subreddit = g.subreddit
    WHERE NOT is_mbti_related
  ),
  x AS (
    SELECT mbti,
      subreddit_group,
      COUNT(*) AS users_per_subreddit_by_mbti
    FROM distinct_users
    GROUP BY mbti,
      subreddit_group
  ),
  x_tot AS (
    SELECT mbti,
      SUM(users_per_subreddit_by_mbti) AS total_users_by_mbti
    FROM x
    GROUP BY mbti
  ),
  x_f AS (
    SELECT x.mbti,
      x.subreddit_group,
      total_users_by_mbti,
      users_per_subreddit_by_mbti,
      ROUND(
        users_per_subreddit_by_mbti * 1.0 / total_users_by_mbti * 1.0,
        4
      ) AS share
    FROM x
      INNER JOIN x_tot ON x.mbti = x_tot.mbti
  )
  SELECT *
  FROM x_f
);