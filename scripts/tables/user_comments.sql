-- Create aggregation table of comments summarized per user
CREATE TABLE IF NOT EXISTS user_comments AS
SELECT 
  author, 
  type, 
  COUNT(*) AS comment_count,
  SUM(score) AS total_comment_score,
  SUM(controversiality) AS total_controversiality
FROM typed_comments
WHERE author IS NOT NULL
GROUP BY author, type;

CREATE INDEX IF NOT EXISTS user_comments_type ON user_comments(type);
CREATE INDEX IF NOT EXISTS user_comments_comment_count ON user_comments(comment_count);
CREATE INDEX IF NOT EXISTS user_comments_total_comment_score ON user_comments(total_comment_score);
CREATE INDEX IF NOT EXISTS user_comments_controversiality ON user_comments(total_controversiality);