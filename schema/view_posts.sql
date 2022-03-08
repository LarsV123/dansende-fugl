-- Simplified view of typed_posts
DROP VIEW IF EXISTS posts;
CREATE VIEW posts AS
SELECT created_utc,
  author,
  mbti_type AS mbti,
  subreddit,
  score,
  stickied,
  gilded,
  distinguished,
  is_self,
  over_18,
  title,
  selftext,
  num_comments,
  permalink
FROM typed_posts;