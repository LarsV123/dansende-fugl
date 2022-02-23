CREATE INDEX IF NOT EXISTS posts_author ON typed_posts(author);
CREATE INDEX IF NOT EXISTS posts_subreddit_id ON typed_posts(subreddit_id);
CREATE INDEX IF NOT EXISTS posts_subreddit ON typed_posts(subreddit);
CREATE INDEX IF NOT EXISTS posts_mbti_type ON typed_posts(type);
CREATE INDEX IF NOT EXISTS posts_created_at ON typed_posts(created_utc);