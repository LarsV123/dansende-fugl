-- Schema for the raw data from typed_comments.csv
-- Columns prefixed with trash_ are not in use, but it makes the
-- insertion easier if we drop them after the data is inserted.
-- Note: Some columns have their names changed compared to the CSV
DROP TABLE IF EXISTS typed_comments CASCADE;
CREATE TABLE typed_comments (
  trash_name TEXT,
  author TEXT NOT NULL,
  author_flair_text TEXT,
  trash_downs TEXT,
  created_utc FLOAT NOT NULL,
  trash_subreddit_id TEXT,
  trash_link_id TEXT,
  trash_parent_id TEXT,
  score FLOAT NOT NULL,
  controversiality FLOAT,
  gilded FLOAT,
  trash_id TEXT,
  subreddit TEXT NOT NULL,
  trash_ups TEXT,
  mbti_type VARCHAR(4),
  word_count INTEGER,
  word_count_quoteless INTEGER,
  quote_to_text_ratio TEXT,
  is_mbti_related BOOLEAN,
  comment TEXT NOT NULL,
  lang VARCHAR(2) NOT NULL
);