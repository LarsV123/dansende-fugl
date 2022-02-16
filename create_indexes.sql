CREATE INDEX comments_author ON typed_comments(author);
CREATE INDEX comments_subreddit_id ON typed_comments(subreddit_id);
CREATE INDEX comments_subreddit ON typed_comments(subreddit);
CREATE INDEX comments_lang ON typed_comments(lang);
CREATE INDEX comments_created_at ON typed_comments(created_utc);
CREATE INDEX comments_controversial ON typed_comments(controversiality);
CREATE INDEX comments_controversial_types ON typed_comments(type,controversiality);

CREATE INDEX posts_author ON typed_posts(author);
CREATE INDEX posts_subreddit_id ON typed_posts(subreddit_id);
CREATE INDEX posts_subreddit ON typed_posts(subreddit);
CREATE INDEX posts_mbti_type ON typed_posts(type);
CREATE INDEX posts_created_at ON typed_posts(created_utc);

CREATE INDEX mbti9k_author ON mbti9k_comments(author);
CREATE INDEX mbti9k_type ON mbti9k_comments(type);
CREATE INDEX mbti9k_comments_num ON mbti9k_comments(comments_num);
