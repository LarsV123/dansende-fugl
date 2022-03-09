-- Simplified view of typed_comments
DROP VIEW IF EXISTS comments;
CREATE VIEW comments AS
SELECT created_utc,
  author,
  mbti_type AS mbti,
  subreddit,
  score,
  controversiality,
  gilded,
  word_count,
  word_count_quoteless,
  quote_to_text_ratio,
  is_mbti_related,
  comment,
  lang
FROM typed_comments;