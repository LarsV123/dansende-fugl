-- Comments per MBTI type
DROP TABLE IF EXISTS comments_per_mbti;
CREATE TABLE comments_per_mbti AS
SELECT mbti,
  COUNT(*) AS mbti_comment_count,
  COUNT(*) * 1.0 / (
    SELECT COUNT(*)
    FROM unique_comments
  ) AS share
FROM unique_comments
GROUP BY mbti
ORDER BY mbti DESC;
-- Unique version
DROP TABLE IF EXISTS unique_comments_per_mbti;
CREATE TABLE unique_comments_per_mbti AS
SELECT mbti,
  COUNT(*) AS mbti_comment_count,
  COUNT(*) * 1.0 / (
    SELECT COUNT(*)
    FROM unique_comments
  ) AS share
FROM unique_comments
GROUP BY mbti
ORDER BY mbti DESC;
-- Posts per MBTI type
DROP TABLE IF EXISTS posts_per_mbti;
CREATE TABLE posts_per_mbti AS
SELECT mbti,
  COUNT(*) AS mbti_post_count
FROM posts
GROUP BY mbti;
-- Comments per hour per type
DROP TABLE IF EXISTS comments_per_hour;
CREATE TEMP TABLE total_comments_per_mbti AS (
  SELECT mbti,
    COUNT(*) AS total_comments
  FROM unique_comments
  GROUP BY mbti
);
CREATE TEMP TABLE temp_comments_per_hour AS (
  SELECT mbti,
    EXTRACT(
      HOUR
      FROM to_timestamp(created_utc)
    ) AS hour,
    COUNT(*) AS comments_per_hour
  FROM unique_comments
  GROUP BY mbti,
    hour
);
CREATE TABLE comments_per_hour AS (
  SELECT mbti,
    comments_per_hour,
    total_comments,
    hour,
    comments_per_hour * 1.0 / total_comments * 1.0 AS share
  FROM total_comments_per_mbti AS a
    NATURAL JOIN temp_comments_per_hour
);
DROP TABLE IF EXISTS top_subreddits_mbti;
CREATE TABLE top_subreddits_mbti AS (
  SELECT subreddit,
    COUNT(*) AS comment_count
  FROM unique_comments
  GROUP BY subreddit
  ORDER BY comment_count DESC
  LIMIT 25
);
DROP TABLE IF EXISTS top_subreddits CASCADE;
CREATE TABLE top_subreddits AS (
  SELECT subreddit,
    COUNT(*) AS comment_count
  FROM unique_comments
  WHERE NOT is_mbti_related
  GROUP BY subreddit
  ORDER BY comment_count DESC
  LIMIT 25
);
DROP TABLE IF EXISTS top_unique_subreddits CASCADE;
CREATE TABLE top_unique_subreddits AS (
  SELECT subreddit,
    COUNT(*) AS comment_count
  FROM unique_comments
  WHERE NOT is_mbti_related
  GROUP BY subreddit
  ORDER BY comment_count DESC
  LIMIT 25
);
DROP VIEW IF EXISTS top_comments;
CREATE VIEW top_comments AS (
  SELECT *
  FROM top_subreddits
    NATURAL JOIN (
      SELECT subreddit,
        mbti,
        COUNT(*) AS mbti_comments
      FROM unique_comments
      WHERE subreddit IN (
          SELECT subreddit
          FROM top_subreddits
        )
      GROUP BY subreddit,
        mbti
    ) AS foo
  ORDER BY comment_count DESC
);
-- Comment count per subreddit and MBTI type
DROP TABLE IF EXISTS mbti_per_subreddit;
CREATE TABLE mbti_per_subreddit AS
SELECT subreddit,
  mbti,
  COUNT(*) AS comment_count
FROM unique_comments
WHERE subreddit IN (
    SELECT subreddit
    FROM top_unique_subreddits
  )
GROUP BY subreddit,
  mbti;
-- Comment count per user and subreddit
DROP TABLE IF EXISTS user_comments_per_subreddit;
CREATE TABLE user_comments_per_subreddit AS
SELECT author,
  subreddit,
  mbti,
  COUNT(*) AS comment_count_by_user
FROM unique_comments
WHERE subreddit IN (
    SELECT subreddit
    FROM top_unique_subreddits
  )
GROUP BY author,
  subreddit,
  mbti;
-- User statistics per subreddit
DROP TABLE IF EXISTS users_per_subreddit;
CREATE TABLE users_per_subreddit AS
SELECT subreddit,
  mbti,
  MAX(comment_count_by_user) AS max_user_comment_count,
  PERCENTILE_DISC(0.5) WITHIN GROUP (
    ORDER BY comment_count_by_user
  ) AS median_comment_count,
  PERCENTILE_DISC(0.9) WITHIN GROUP (
    ORDER BY comment_count_by_user
  ) AS percentile_90_comment_count,
  PERCENTILE_DISC(0.95) WITHIN GROUP (
    ORDER BY comment_count_by_user
  ) AS percentile_95_comment_count,
  ROUND(AVG(comment_count_by_user)) AS average_comment_count,
  SUM(comment_count_by_user) AS total_comment_count,
  COUNT(*) AS distinct_users
FROM user_comments_per_subreddit
GROUP BY subreddit,
  mbti
ORDER BY subreddit,
  mbti;