CREATE INDEX author ON typed_comments(author);
CREATE INDEX subreddit_id ON typed_comments(subreddit_id);
CREATE INDEX subreddit ON typed_comments(subreddit);
CREATE INDEX lang ON typed_comments(lang);
CREATE INDEX created_at ON typed_comments(created_utc);

CREATE INDEX author ON typed_posts(author);
CREATE INDEX subreddit_id ON typed_posts(subreddit_id);
CREATE INDEX subreddit ON typed_posts(subreddit);
CREATE INDEX mbti_type ON typed_posts(type);
CREATE INDEX created_at ON typed_posts(created_utc);

CREATE INDEX author ON mbti9k_comments(author);
CREATE INDEX mbti_type ON mbti9k_comments(type);
CREATE INDEX comments_num ON mbti9k_comments(comments_num);
