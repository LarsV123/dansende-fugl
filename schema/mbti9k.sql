-- Schema for the raw data from mbti9k.csv
-- Columns prefixed with trash_ are not in use, but it makes the
-- insertion easier if we drop them after the data is inserted.
-- Note: Some columns have their names changed compared to the CSV
DROP TABLE IF EXISTS mbti9k CASCADE;
CREATE TABLE mbti9k (
  author TEXT NOT NULL,
  comment TEXT NOT NULL,
  mbti_type VARCHAR(4) NOT NULL,
  subreddits_commented INTEGER NOT NULL,
  mbti_subreddits_commented INTEGER NOT NULL,
  wc INTEGER NOT NULL,
  comments_num INTEGER NOT NULL
);