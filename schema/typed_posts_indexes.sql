-- Single column indexes
CREATE INDEX IF NOT EXISTS posts_created_utc ON typed_posts(created_utc);
CREATE INDEX IF NOT EXISTS posts_subreddit ON typed_posts(subreddit);
CREATE INDEX IF NOT EXISTS posts_author ON typed_posts(author);
CREATE INDEX IF NOT EXISTS posts_domain ON typed_posts(domain);
CREATE INDEX IF NOT EXISTS posts_num_comments ON typed_posts(num_comments);
CREATE INDEX IF NOT EXISTS posts_score ON typed_posts(score);
CREATE INDEX IF NOT EXISTS posts_gilded ON typed_posts(gilded);
CREATE INDEX IF NOT EXISTS posts_stickied ON typed_posts(stickied);
CREATE INDEX IF NOT EXISTS posts_retrieved_on ON typed_posts(retrieved_on);
CREATE INDEX IF NOT EXISTS posts_over_18 ON typed_posts(over_18)
WHERE over_18;
CREATE INDEX IF NOT EXISTS posts_is_self ON typed_posts(is_self);
CREATE INDEX IF NOT EXISTS posts_distinguished ON typed_posts(distinguished);
CREATE INDEX IF NOT EXISTS posts_mbti_type ON typed_posts(mbti_type);
--Compound indexes
CREATE INDEX IF NOT EXISTS posts_type_to_subreddit ON typed_posts(subreddit, mbti_type);
CREATE INDEX IF NOT EXISTS posts_type_to_author ON typed_posts(subreddit, author);