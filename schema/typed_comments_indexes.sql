-- Single column indexes
CREATE INDEX IF NOT EXISTS comments_author ON typed_comments(author);
CREATE INDEX IF NOT EXISTS comments_created_utc ON typed_comments(created_utc);
CREATE INDEX IF NOT EXISTS comments_score ON typed_comments(score);
CREATE INDEX IF NOT EXISTS comments_controversiality ON typed_comments(controversiality);
CREATE INDEX IF NOT EXISTS comments_gilded ON typed_comments(gilded);
CREATE INDEX IF NOT EXISTS comments_subreddit ON typed_comments(subreddit);
CREATE INDEX IF NOT EXISTS comments_mbti_type ON typed_comments(mbti_type);
CREATE INDEX IF NOT EXISTS comments_word_count ON typed_comments(word_count);
CREATE INDEX IF NOT EXISTS comments_word_count_quoteless ON typed_comments(word_count_quoteless);
CREATE INDEX IF NOT EXISTS comments_quote_to_text_ratio ON typed_comments(quote_to_text_ratio);
CREATE INDEX IF NOT EXISTS comments_is_mbti_related ON typed_comments(is_mbti_related);
CREATE INDEX IF NOT EXISTS comments_lang ON typed_comments(lang);
-- Compound indexes
CREATE INDEX IF NOT EXISTS comments_type_to_subreddit ON typed_comments(mbti_type, subreddit);
CREATE INDEX IF NOT EXISTS comments_type_to_author ON typed_comments(mbti_type, author);
CREATE INDEX IF NOT EXISTS comments_controversiality_to_typoe ON typed_comments(controversiality, mbti_type);