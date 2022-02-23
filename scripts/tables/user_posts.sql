-- Create aggregation table of posts summarized per user
DROP TABLE IF EXISTS user_post_summary;

CREATE TABLE user_post_summary AS
SELECT author, type, SUM(score) AS score, SUM(num_comments) AS num_comments, COUNT(*) AS post_count
FROM typed_posts
WHERE author IS NOT NULL
GROUP BY author;

CREATE UNIQUE INDEX IF NOT EXISTS user_post_summary_author ON user_post_summary(author);
CREATE INDEX IF NOT EXISTS user_post_summary_type ON user_post_summary(type);
CREATE INDEX IF NOT EXISTS user_post_summary_post_count ON user_post_summary(post_count);