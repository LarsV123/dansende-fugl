-- Schema for the raw data from typed_posts.csv
-- Columns prefixed with trash_ are not in use, but it makes the
-- insertion easier if we drop them after the data is inserted.
-- Note: Some columns have their names changed compared to the CSV
DROP TABLE IF EXISTS typed_posts CASCADE;
CREATE TABLE typed_posts (
  created_utc INTEGER NOT NULL,
  subreddit TEXT NOT NULL,
  author TEXT NOT NULL,
  domain TEXT NOT NULL,
  link_url TEXT NOT NULL,
  num_comments INTEGER,
  score INTEGER NOT NULL,
  trash_ups TEXT,
  trash_downs TEXT,
  title TEXT NOT NULL,
  selftext TEXT,
  trash_saved TEXT,
  trash_id TEXT,
  trash_from_kind TEXT,
  gilded INTEGER,
  trash_from TEXT,
  stickied BOOLEAN,
  retrieved_on INTEGER,
  over_18 BOOLEAN,
  thumbnail TEXT,
  trash_subreddit_id TEXT,
  trash_hide_score TEXT,
  link_flair_css_class TEXT,
  author_flair_css_class TEXT,
  trash_archived TEXT,
  is_self BOOLEAN,
  trash_from_id TEXT,
  permalink TEXT,
  trash_name TEXT,
  author_flair_text TEXT,
  trash_quarantine TEXT,
  link_flair_text TEXT,
  distinguished TEXT,
  mbti_type VARCHAR(4)
);