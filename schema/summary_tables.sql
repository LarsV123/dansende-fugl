-- Comments per MBTI type
DROP TABLE IF EXISTS comments_per_mbti;
CREATE TABLE comments_per_mbti AS
SELECT mbti,
  COUNT(*) AS mbti_comment_count
FROM comments
GROUP BY mbti;
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
  FROM comments
  GROUP BY mbti
);
CREATE TEMP TABLE temp_comments_per_hour AS (
  SELECT mbti,
    EXTRACT(
      HOUR
      FROM to_timestamp(created_utc)
    ) AS hour,
    COUNT(*) AS comments_per_hour
  FROM comments
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
  FROM comments
  GROUP BY subreddit
  ORDER BY comment_count DESC
  LIMIT 25
);
DROP TABLE IF EXISTS top_subreddits CASCADE;
CREATE TABLE top_subreddits AS (
  SELECT subreddit,
    COUNT(*) AS comment_count
  FROM comments
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
      FROM comments
      WHERE subreddit IN (
          SELECT subreddit
          FROM top_subreddits
        )
      GROUP BY subreddit,
        mbti
    ) AS foo
  ORDER BY comment_count DESC
);
DROP VIEW IF EXISTS heatmap;
CREATE VIEW heatmap AS (
  SELECT *
  FROM top_subreddits
    NATURAL JOIN (
      SELECT subreddit,
        mbti,
        COUNT(*) AS mbti_comments
      FROM comments
      GROUP BY subreddit,
        mbti
    ) AS foo
  ORDER BY comment_count DESC
);