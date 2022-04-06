-- Find count of comments for each users in each subreddit
DROP TABLE IF EXISTS user_comments_per_subreddit;
CREATE TABLE user_comments_per_subreddit AS
SELECT author,
  subreddit,
  COUNT(*) AS comment_count_by_user
FROM unique_comments
WHERE NOT is_mbti_related
GROUP BY author,
  subreddit;
-- Find top subreddits by count of distinct users
DROP TABLE IF EXISTS top_subreddits_by_distinct_users;
CREATE TABLE top_subreddits_by_distinct_users AS (
  SELECT subreddit,
    COUNT(*) AS distinct_users
  FROM user_comments_per_subreddit
  GROUP BY subreddit
  ORDER BY distinct_users DESC
);
-- Find top subreddits by MBIT and count of distinct users
DROP TABLE IF EXISTS top_subreddits_by_mbti_distinct_users;
CREATE TABLE top_subreddits_by_mbti_distinct_users AS (
  SELECT subreddit,
    mbti,
    COUNT(*) AS distinct_users
  FROM unique_comments
  WHERE subreddit IN (
      SELECT subreddit
      FROM top_subreddits_by_distinct_users
      LIMIT 25
    )
  GROUP BY subreddit,
    mbti
  ORDER BY distinct_users DESC
);
-- Find count of MBTI types for top subreddits
DROP TABLE IF EXISTS mbti_per_subreddit;
CREATE TABLE mbti_per_subreddit AS
SELECT subreddit,
  mbti,
  COUNT(*) AS comment_count
FROM unique_comments
WHERE subreddit IN (
    SELECT subreddit
    FROM top_subreddits_by_distinct_users
    LIMIT 25
  )
GROUP BY subreddit,
  mbti;
-- Find comment count per user per top subreddit
DROP TABLE IF EXISTS user_comments_per_subreddit;
CREATE TABLE user_comments_per_subreddit AS
SELECT author,
  subreddit,
  mbti,
  COUNT(*) AS comment_count_by_user
FROM unique_comments
WHERE subreddit IN (
    SELECT subreddit
    FROM top_subreddits_by_distinct_users
    LIMIT 25
  )
GROUP BY author,
  subreddit,
  mbti;