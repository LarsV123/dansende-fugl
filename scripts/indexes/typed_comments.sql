CREATE INDEX IF NOT EXISTS comments_author ON typed_comments(author);
CREATE INDEX IF NOT EXISTS comments_subreddit_id ON typed_comments(subreddit_id);
CREATE INDEX IF NOT EXISTS comments_subreddit ON typed_comments(subreddit);
CREATE INDEX IF NOT EXISTS comments_lang ON typed_comments(lang);
CREATE INDEX IF NOT EXISTS comments_created_at ON typed_comments(created_utc);
CREATE INDEX IF NOT EXISTS comments_controversial ON typed_comments(controversiality);
CREATE INDEX IF NOT EXISTS comments_controversial_types ON typed_comments(type,controversiality);