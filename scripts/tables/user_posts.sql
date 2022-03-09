-- Create aggregation table of posts summarized per user
CREATE TABLE IF NOT EXISTS user_posts AS
SELECT 
  author, 
  type, 
  COUNT(*) AS post_count,
  SUM(score) AS total_post_score, 
  SUM(num_comments) AS total_comments_received
FROM typed_posts
WHERE author IS NOT NULL
GROUP BY author;

CREATE UNIQUE INDEX IF NOT EXISTS user_posts_author ON user_posts(author);
CREATE INDEX IF NOT EXISTS user_posts_type ON user_posts(type);
CREATE INDEX IF NOT EXISTS user_posts_post_count ON user_posts(post_count);
CREATE INDEX IF NOT EXISTS user_posts_total_post_score ON user_posts(total_post_score);
CREATE INDEX IF NOT EXISTS user_posts_total_comments_received ON user_posts(total_comments_received);